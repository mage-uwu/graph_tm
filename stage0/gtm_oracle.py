"""
gtm_oracle: slow, obviously-correct reference for the Graph Tsetlin Machine.

Transcribed from cair/GraphTsetlinMachine (kernels.py + tm.py). Mapping:

  calculate_messages             -> _eval, layer 0
  prepare/exchange/encode_msgs   -> _messages
  calculate_messages_conditional -> _eval, layers 1..D-1
  evaluate                       -> class_sum over clauses true on any node
  select_clause_node             -> _select_nodes
  select_clause_updates          -> _select_updates (also updates weights)
  update / update_message        -> _update_tas (Type I / Type II feedback)

Deliberate differences from the CUDA code, none of which change semantics:
  * RNG: counter-based (gtmcore.key) instead of curand, so runs are reproducible.
  * TA states are plain ints, not bit planes. inc/dec saturate at [0, 2^B - 1],
    which is exactly what the bit-sliced carry logic does.
  * Padding literals past L are ignored instead of carried along.
"""
import numpy as np
from gtmcore import (ModelConfig, key, prob_threshold16, feedback_mask, clause_hypervectors, rank_plan,
                     TAG_INIT_W, TAG_NODE_SEL, update_prob, selection_masks, save_model, load_model)

INT_MAX = 2**31 - 1


class OracleGTM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        c = cfg
        self.half = 1 << (c.B - 1)
        self.maxs = (1 << c.B) - 1
        # prepare(): weights +-1 at random (negative clauses) else 1
        if c.neg:
            bits = np.array([[int(key(c.seed, TAG_INIT_W, k, j, 0)) & 1 for j in range(c.C)]
                             for k in range(c.O)], dtype=np.int64)
            self.w = 1 - 2 * bits
        else:
            self.w = np.ones((c.O, c.C), dtype=np.int64)
        # TA states: all planes but MSB set -> 2^(B-1) - 1 (just below include)
        self.ta = [np.full((c.C, c.L), self.half - 1, dtype=np.int32)]
        for _ in range(c.D - 1):
            self.ta.append(np.full((c.C, c.M), self.half - 1, dtype=np.int32))
        self.hv = clause_hypervectors(c.seed, c.C, c.MS, c.MB)
        self.step = 0
        self.thr_s = [prob_threshold16(1.0 / s) for s in c.s]
        self.rank = None  # training option, not persisted: (negative table (Nn, O) uint8, K, margin)

    # ------------------------------------------------------------------ io
    def save(self, path):
        save_model(path, self.cfg, self.hv, self.w, self.ta, self.step)

    @staticmethod
    def load(path):
        cfg, hv, w, ta, step = load_model(path)
        m = OracleGTM.__new__(OracleGTM)
        m.cfg, m.hv, m.w, m.ta, m.step = cfg, hv, w, ta, step
        m.half, m.maxs = 1 << (cfg.B - 1), (1 << cfg.B) - 1
        m.thr_s = [prob_threshold16(1.0 / s) for s in cfg.s]
        m.rank = None
        return m

    # ------------------------------------------------------------ evaluate
    def _messages(self, ds, n0, n, out):
        """exchange_messages: every clause true at a source node sends its hypervector,
        rotated by edge type, to each destination. Bundled with OR.
        senders=1: only clauses that include a positive layer-0 literal send."""
        c = self.cfg
        pos = np.zeros((n, c.MS), dtype=bool)
        can_send = (self.ta[0][:, :c.H] >= self.half).any(1) if c.senders else np.ones(c.C, dtype=bool)
        for src in range(n):
            fired = np.nonzero(out[:, src] & can_send)[0]
            if len(fired) == 0:
                continue
            e0, e1 = ds.edge_offset[n0 + src], ds.edge_offset[n0 + src + 1]
            for dst, et in ds.edges[e0:e1]:
                bits = (self.hv[fired].astype(np.int64) + int(et)) % c.MS
                pos[dst, bits.ravel()] = True
        return np.concatenate([pos, ~pos], axis=1)  # [m | not m]

    def _eval(self, ds, g):
        c = self.cfg
        n0, n1 = ds.node_offset[g], ds.node_offset[g + 1]
        n = int(n1 - n0)
        X0 = ds.X[n0:n1]
        types = ds.node_type[n0:n1]
        typeok = types[None, :] == (np.arange(c.C) % ds.n_node_types)[:, None]

        inc = self.ta[0] >= self.half
        viol = (inc.astype(np.int32) @ (~X0).astype(np.int32).T) > 0
        out = typeok & ~viol
        n_inc = inc.sum(1)
        layer_X = [X0]
        alive = [out.any(axis=1)]
        for d in range(1, c.D):
            Xm = self._messages(ds, n0, n, out)
            inc = self.ta[d] >= self.half
            viol = (inc.astype(np.int32) @ (~Xm).astype(np.int32).T) > 0
            out = out & typeok & ~viol
            n_inc = n_inc + inc.sum(1)
            layer_X.append(Xm)
            alive.append(out.any(axis=1))
        clause_true = out.any(axis=1)
        class_sum = self.w[:, clause_true].sum(axis=1)
        # first layer at which each clause is false at every node (D if it survives all layers)
        alive = np.stack(alive)
        dead_at = np.where(alive.all(0), c.D, np.argmin(alive, axis=0))
        self._dead_at = dead_at
        return out, clause_true, class_sum, n_inc, layer_X

    def score(self, ds):
        return np.stack([self._eval(ds, g)[2] for g in range(ds.n_graphs)])

    def transform(self, ds):
        return np.stack([self._eval(ds, g)[1] for g in range(ds.n_graphs)])

    # --------------------------------------------------------------- train
    def _select_nodes(self, out, t):
        sel = np.full(self.cfg.C, -1, dtype=np.int64)
        for j in range(self.cfg.C):
            nodes = np.nonzero(out[j])[0]
            if len(nodes):
                sel[j] = nodes[int(key(self.cfg.seed, TAG_NODE_SEL, t, j, 0)) % len(nodes)]
        return sel

    def _select_updates(self, class_sum, yenc, sel, t, ybits=None):
        """select_clause_updates: per output, a Bernoulli(err/2T * q') subset of clauses gets
        feedback of sign target * sign(weight); weights of fired clauses move accordingly.
        Returns the automaton feedback signs: a rho-subset of the same pairs (decoupled feedback)."""
        c = self.cfg
        upd = np.zeros((c.O, c.C), dtype=np.int64)
        cs = np.clip(class_sum, -c.T, c.T)
        fired = sel != -1
        rp = None
        if self.rank is not None:  # (negative table, K, margin): ranking feedback, see rank_plan
            rp = rank_plan(c.seed, t, cs.astype(np.int64), ybits, *self.rank)
            if rp is None:
                return upd
        for k in range(c.O):
            if rp is not None:
                if not rp[0][k]:
                    continue
                target, p = (1 if ybits[k] else -1), rp[1]
            else:
                y = int(yenc[k])
                target = 1 - 2 * int(cs[k] > y)
                p = update_prob(abs(y - int(cs[k])), c.T, target, c.q, c.O)
            chosen, chosen_ta = selection_masks(c.seed, t, k, c.C, prob_threshold16(p), prob_threshold16(p * c.rho))
            if not chosen.any():
                continue
            w = self.w[k]
            sign = np.where(w >= 0, 1, -1)
            ts = target * sign
            upd[k] = np.where(chosen_ta, ts, 0)
            grow = chosen & fired & (ts > 0) & (np.abs(w) < INT_MAX)
            shrink = chosen & fired & (ts < 0)
            w[grow] += sign[grow]
            w[shrink] -= sign[shrink]
            if not c.neg:
                w[shrink & (w < 1)] = 1
        return upd

    def _update_tas(self, upd, sel, n_inc, layer_X, t):
        c = self.cfg
        for layer in range(c.D):
            ta = self.ta[layer]
            nl = ta.shape[1]
            for j in range(c.C):
                node = sel[j]
                fired = node != -1
                Xc = layer_X[layer][node] if fired else None
                for k in range(c.O):
                    fb_sign = upd[k, j]
                    if fb_sign > 0:  # Type I
                        if not fired and c.layered and layer < self._dead_at[j]:
                            continue  # layered forget: this layer matched somewhere, spare it
                        fb = feedback_mask(c.seed, t, j, k, c.O, layer, nl, self.thr_s[layer])
                        st = ta[j]
                        if fired and n_inc[j] <= c.max_inc:
                            inc_m = Xc if c.boost else (Xc & ~fb)
                            dec_m = ~Xc & fb
                            st[inc_m] = np.minimum(st[inc_m] + 1, self.maxs)
                            st[dec_m] = np.maximum(st[dec_m] - 1, 0)
                        else:
                            st[fb] = np.maximum(st[fb] - 1, 0)
                    elif fb_sign < 0 and fired:  # Type II
                        st = ta[j]
                        m = ~Xc & (st < self.half)
                        st[m] = st[m] + 1

    def fit_one(self, ds, g, yenc):
        out, _, class_sum, n_inc, layer_X = self._eval(ds, g)
        t = self.step
        sel = self._select_nodes(out, t)
        upd = self._select_updates(class_sum, yenc, sel, t, (yenc > 0).astype(np.uint8))
        self._update_tas(upd, sel, n_inc, layer_X, t)
        self.step += 1

    def fit(self, ds, epochs=1, max_steps=None):
        yenc = np.where(ds.Y == 1, self.cfg.T, -self.cfg.T)
        done = 0
        for _ in range(epochs):
            for g in range(ds.n_graphs):
                if max_steps is not None and done >= max_steps:
                    return
                self.fit_one(ds, g, yenc[g])
                done += 1

    def accuracy(self, ds):
        sums = self.score(ds)
        if ds.task_kind == 0:
            return float((sums.argmax(1) == ds.Y.argmax(1)).mean())
        return float(((sums >= 0).astype(int) == ds.Y).all(1).mean())
