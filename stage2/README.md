# Stage 2: real-text pretraining + BERT-style adapters

Question: does a ~4.4M-parameter GraphTM (bert-tiny scale and depth), pretrained as a masked
autoencoder on real text, learn representations that transfer to classic BERT tasks?

`./run_stage2.sh` runs everything (data, pretraining, controls, adapters, bert-tiny baselines)
and prints one `RESULT {json}` line per measurement. No engine changes: this stage only writes
`.gtmd` files and calls `stage1/gtm`.

## Pretraining

| | |
|---|---|
| Corpus | WikiText-103 (raw), paragraphs, headings dropped: 111.5M WordPiece tokens |
| Tokenizer | bert-base-uncased WordPiece (bert-tiny's vocabulary, 30,522) |
| Example | a +-8-token window around one position, that position replaced by `[MASK]` |
| Graph | one node per token; edges at distance 1, 2, 4 on each side (6 edge types); depth 3 = 2 hops = +-8 tokens |
| Node features | a fixed random dense code per token id (64 of 128 bits), literals `[x | not x]` |
| Node types | centre = 1, context = 0: odd clauses fire only at `[MASK]`, even clauses are context detectors |
| Target | 256 LSH sign bits of bert-base-uncased's input embedding of the hidden token (teacher); no teacher information in the input |
| Model | C=2880, H=128, msg 256, depth 3, O=256: 2880 x (256 + 2x512 automata + 256 weights) = 4.42M |
| Config | `--T 50000 --s 5 --senders pos --q -1 --rho 2/256` |
| Eval | rank of the true token among 29,522 real vocab tokens by vote-weighted code agreement: acc@1, acc@10, MRR; per-bit accuracy; same windows scored by bert-tiny's own MLM head |

## Adapters (frozen GraphTM)

| Task | Shape | GraphTM features |
|---|---|---|
| SST-2 | single sentence, 2 classes | mask each token in turn, OR-pool the clause bits over the sentence |
| QNLI | sentence pair, 2 classes | [pool(question), pool(sentence), AND of both] |
| CoNLL-2003 NER | token tagging, 9 BIO tags, entity F1 | clause bits of the graph that masks the word (context only) |

Heads: a logistic-regression probe and a TM head (depth-1 GraphTM on one node), both tuned on
dev. Comparisons: lexical bag-of-token-ids baseline, GraphTM + lexical, a depth-1 pretrained
control (bag of context, no message passing), bert-tiny frozen (+ the same logistic probe)
and bert-tiny fully fine-tuned. GLUE test labels are hidden, so GLUE validation is the test set
and 2,000 held-out training examples are the dev set.

## Findings while building it (200k-window runs, WikiText-103 validation)

1. **Sparse token codes teach the TM "NOT" clauses.** With 3-of-128-bit codes nearly every
   negated literal is true for nearly every token; after 300k windows clauses held ~47 negated
   and 0.03 positive layer-0 includes, fired on ~every window, and per-bit accuracy drifted
   *down* (0.637 -> 0.574) below the unigram prior (0.689). Dense codes (64 of 128) keep
   negation meaningful: each literal is true for half the vocabulary.
2. **OR over 17 nodes makes every clause a bias term.** Giving the centre its own node type
   restricts half the clauses to the masked position, where they must use messages.
3. **T must scale with the clauses that fire, not C/4.** With 2,880 clauses firing on every
   example, one weight update moves a vote sum by ~1,000, so at T=720 the sum overshoots
   every step. With a context-free input (only the `[MASK]` node, best possible = the
   frequency prior, 0.687 per bit): T=720 -> 0.613, T=5,000 -> 0.672, T=20,000 -> 0.686.
   The toy tasks never showed this because their clauses fired sparsely.

After all three: per-bit 0.692 (prior 0.689), acc@1 0.065 (unigram 0.060), acc@10 0.245
(unigram 0.277) at 200k windows. bert-tiny's MLM head on the same windows: acc@1 0.302,
acc@10 0.569.

## Status after the first full run (negative result, recorded as found)

Pod run (16 threads, 3.0-5.3k windows/s) with the config above: validation acc@10 fell with
data, 0.238 (1M windows) -> 0.224 (2M) -> 0.220 (3M) -> 0.214 (4M) -> 0.211 (5M), per-bit
0.689 -> 0.666, message fill 77% -> 58%. The best point is ~200k windows; nothing learned
beyond ~1.5 points of acc@10 over a context-free model.

Local diagnosis (clause statistics per 100k windows):

4. **The message layers repeat finding 1, mirrored.** Centre clauses include ~96 positive and
   0 negated message literals. At 80-88% fill a set bit is nearly always true (cheap to
   include, uninformative) and an unset bit is true ~17% of the time, below the ~1/s = 20%
   inclusion threshold at s=5, so the informative "bit is 0" literals are never learned.
   Context clauses slowly specialise (7.7 -> 14.2 layer-0 includes over 700k windows), fewer
   clauses send, fill drops, the centre clauses' 96-bit conjunctions stop matching, and
   accuracy drifts down while weights grow without bound.

Tried, did not help (400k windows each, WikiText-103 validation acc@10; prior-only 0.231):

| change | fill | acc@10 |
|---|---|---|
| baseline config | 80-88% | 0.235-0.247 |
| `--msg-bits 1` (fill is set by in-degree 6 x firing context clauses, not bits per clause) | 83-85% | 0.236-0.243 |
| `--s 5,25,25` (per-layer s): message clauses over-specialise, nothing fires, fill 0 | 0% | 0.231 (= prior) |
| msg-size 1024, C=960 (same params) | 18% | 0.230-0.235 |

Full-run curve (validation, 10k windows; context-free prior per-bit 0.689, acc@10 0.231):

| windows | UTC | acc@10 | per-bit | fill |
|---|---|---|---|---|
| 1M | 21:10:06 | 0.2377 | 0.6887 | 77.5% |
| **2M = p-collapse** | **21:14:59** | **0.2237** | **0.6779** | 67.7% |
| 3M | 21:20:42 | 0.2195 | 0.6740 | 62.7% |
| 5M | 21:27:31 | 0.2109 | 0.6663 | 57.8% |
| 10M | 21:38:29 | 0.2279 | 0.6768 | 55.1% |

Test (20k windows): depth 3 acc@10 0.2235 / per-bit 0.6733; the depth-1 control (2M windows, no
message passing) 0.2595 / 0.6874, i.e. message passing was net harmful. SST-2 with the depth-3
clause bits: 0.536 (logistic) / 0.538 (TM head) vs 0.787 for bag of words.

## Testbed

`testbed.sh` (on the pod) + `testbed.py`: experiments queued in `experiments.tsv` run to
p-collapse (2M windows) with validation at 250k/500k/1M/1.5M/2M and clause statistics, so every
change is benchmarked against the same point where the first run failed.

### Testbed results so far (validation at p-collapse = 2M windows; bert-tiny 0.300 / 0.566)

| exp | change | acc@1 | acc@10 | per-bit | centre fill |
|---|---|---|---|---|---|
| E0 | first-run config (depth 3, msg 256, +-1/2/4) | 0.054 | 0.230 | 0.681 | 97-98% |
| E1 | **flat oracle**: depth 1 on the +-2 neighbours' codes | 0.148 | 0.307 | 0.712 | - |
| ridge | linear readout of the same +-2 codes (500k windows) | 0.145 | 0.303 | 0.710 | - |
| R3e | flat oracle on +-8 codes (C=1000, matched params) | 0.136 | 0.290 | 0.706 | - |
| R2b | depth 2, msg 512, 1 bit/sender, +-1/2 | 0.122 | 0.281 | 0.702 | 88% |
| **R3d** | R2b with s=10 | **0.130** | **0.285** | **0.707** | 71% |
| R3f | R2b with s=3 | 0.120 | 0.278 (1M) | 0.701 | 90% |
| R3a | R2b with +-4 edges too | 0.106 (1M) | 0.275 | 0.700 | 88% |
| R2a/R2c | empty [MASK] code | 0.053 | 0.229 | 0.687 | 8-11% |
| R2d | depth 2, msg 256, 2 bits | 0.066 | 0.228 | 0.676 | 92% |
| R3b | depth 3 (2 hops), C=1728 | 0.053 | 0.230 | 0.689 | 1% at layer 2 |
| R3c | depth 2, msg 1024, C=1728 | 0.053 | 0.226 | 0.689 | 7% |
| E2 | depth-1 bag of window (control) | 0.063 | 0.250 (500k) | 0.691 | - |

Findings: (a) the TM head is not the limit on +-2 context (E1 = ridge); (b) one hop at msg 512
recovers ~93% of that; (c) extra context does not help even when handed over directly (R3e), so
at this budget a TM over random token codes behaves roughly like a linear model over them, which
caps it far below bert-tiny; (d) every run that collapses (R2a, R2c, R3b, R3c) collapses the same
way: context clauses lose all layer-0 literals, stop sending, and fill dies -> sender autopsy.

### Depth (rounds 4-8)

Sender autopsy (`autopsy.py`, oracle traces per checkpoint): ~72% of a clause's Type I events
arrive while its final conjunction is false everywhere; the "forget" applied then hits every
layer, so a clause failing at a message layer loses its layer-0 (sender) literals. Per layer-0
literal the drift is ~0 at s=5 (+0.028 Type I fired, +0.035 Type II, -0.068 forget), so senders
have no restoring force and depth 3's transient empties them (83% empty layer 0 at 50k). A
literal-budget explanation was tested and refuted (`--max-inc 4096` still collapses).

| exp | change (depth 3 unless noted, C=1728 matched params) | acc@1 | acc@10 @2M | stability |
|---|---|---|---|---|
| R3b | baseline | 0.053 | 0.230 | collapsed by 250k |
| R4a | residual init (message automata start 64 below include) | 0.074 | 0.241 | 0.280 @500k, then collapses |
| R5a/b | per-layer s, layer 0 at 10/20 | ~0.06 | 0.249 | senders over-specialise, fill 2-5% |
| **L1** | **engine: `--forget layered`** | 0.102 | **0.277** | **stable (first stable depth 3)** |
| L2 | layered + residual init | 0.112 | 0.282 | stable |
| R4b | depth 2 + residual init (C=2880) | 0.137 | **0.297** | best overall, 97% of flat oracle |
| L3 | depth 2 + layered + residual init | 0.115 | 0.288 | stable |
| B1 | BLT entropy patches as nodes, depth 2 | 0.121 | 0.275 | still climbing |

Layered forget overshoots: without forget from deeper failures, layer 0 only grows and context
clauses become exact token detectors (~51+51 of 128 literals, 19 senders/node), so layer-2 centre
fill is ~2% and the second hop carries almost nothing. A literal budget bounds this (local trace,
`--max-inc 32`: ~22 layer-0 literals, layer-2 centre fill 7.5%), but round 8 shows it costs more
than it gains: the second hop gets more signal, while the first hop and the centre clauses lose it.

| exp | layered + ri64, depth 3, plus | acc@1 | acc@10 @2M | layer-1 / layer-2 centre fill |
|---|---|---|---|---|
| L2 | - | 0.112 | **0.282** | 11% / 2% |
| L5 | `--max-inc 32` | 0.071 | 0.270 | 9% / 8% |
| L6 | `--max-inc 16` | 0.070 | 0.270 | 9% / 7% |
| L9 | `--max-inc 8` | 0.047 | 0.245 | 10% / 7% |
| B3 | (depth 2) token + hashed-n-gram codes, s=10 | 0.104 | 0.264 | 63% |
| B2 | (flat oracle) BLT patch codes of +-2 tokens | 0.100 | 0.251 (0.265 @500k, drifting) | - |

So a global literal cap is not the fix for layered overshoot, and neither BLT variant beats random
token codes: the byte-level codes carry less token identity than the +-2 token codes they replace.

## Where the gap to bert-tiny comes from (`code_ceiling.py`, `decoder_cap.py`)

Validation acc@10 on the same 10k windows. The GraphTM tracks a linear readout of its input codes
(R4b 0.297 vs ridge 0.306 on random codes), so the ceilings of linear readouts and of the output
decoder bound it:

| what is measured | acc@1 | acc@10 |
|---|---|---|
| R4b (best GraphTM) | 0.137 | 0.297 |
| ridge, +-2 context, random codes (the current input) | 0.145 | 0.306 |
| ridge, +-2, **learned codes**: co-occurrence PPMI -> SVD -> balanced sign bits (gradient free, teacher free) | 0.183 | 0.348 |
| ridge, +-2, sign bits of bert-tiny's own trained embeddings | 0.187 | 0.350 |
| ridge, +-8 context: random / learned / bert-tiny codes | 0.150 / 0.187 / 0.194 | 0.314 / 0.354 / 0.360 |
| bert-tiny's full distribution pushed through our output (256 teacher bits, vote-weighted agreement) | 0.287 | 0.469 |
| same with 256 / 1024 / 4096 random output codes | 0.302 | 0.443 / 0.508 / 0.543 |
| bert-tiny MLM head directly | 0.302 | 0.569 |

So the 0.30 -> 0.57 gap splits into roughly three parts:
1. **Output decoder, ~0.10.** Predicting 256 code bits and ranking by agreement blurs any uncertain
   prediction: even bert-tiny's own distribution only reaches 0.469 through it. Wider output codes
   recover most of it (4096 bits: 0.543), at a cost in output weights per clause.
2. **Token codes, ~0.05.** Learned codes lift the linear ceiling 0.306 -> 0.348, and gradient-free
   co-occurrence codes are as good as bert-tiny's trained embeddings here (0.348 vs 0.350).
   `GTM_CODES=dist` selects them as GraphTM input.
3. **Nonlinear composition, ~0.11** (0.36 -> 0.47): what bert-tiny's layers add over any linear map
   of learned codes. This is the part a TM should supply through clauses and has not yet: on
   random codes it matches the linear readout exactly.

### The output decoder (the largest single piece)

Tying the output to the learned codes does not remove the cap (`output_codes.py`; cap = bert-tiny's
distribution through the decoder, ridge = linear readout of +-2 learned input codes):

| output code | cap acc@10 | ridge acc@10 |
|---|---|---|
| teacher 256 (current) | 0.469 | 0.348 |
| learned 256, tied | 0.461 | 0.350 |
| learned 128 = the input codes | 0.445 | 0.335 |
| learned, continuous decoding (like BERT's tied output) | 0.39-0.42 | 0.25-0.31 |

The decoder already has BERT's form (score = query . code_t). What caps it is the objective: the
TM trains every output bit independently, so its votes converge to the expected code of the
plausible tokens (per-output equilibrium cs_k = T (2 P(bit k) - 1)). Same linear model, same input,
same fixed teacher codes (`query_objective.py`): per-bit regression 0.348, softmax (ranking)
through the codes **0.418**, unconstrained linear softmax 0.480.

Engine: opt-in `--rank K --neg-table F [--rank-margin M]` (oracle `gtmcore.rank_plan`): feedback
only on the outputs where the target code and the best of K sampled negative codes differ, with
the pairwise margin probability. Flat oracle on learned input codes, 750k windows:

| output feedback | acc@1 | acc@10 | MRR | frequent acc@10 |
|---|---|---|---|---|
| per-bit (default) | 0.181 | 0.3425 | 0.237 | 0.724 |
| rank K=16, M=T | 0.117 | 0.222 | 0.153 | 0.429 |
| rank K=16, M=8T | **0.197** | 0.338 | **0.246** | 0.653 |
| rank K=16, M=64T | 0.189 | 0.343 | 0.243 | 0.672 |
| rank K=256, M=8T / 64T | 0.145 / 0.148 | 0.238 / 0.249 | 0.178 / 0.183 | 0.456 / 0.480 |

Pairwise ranking helps the top of the list (acc@1 +1.6, MRR +0.9) but is far from the +7 points
of the linear softmax, and the hardest negative hurts: it is usually a plausible token (a vs the),
pushed down on every example, so frequent tokens lose rank. Softmax instead pushes away from the
model's own expected code, i.e. a negative *sampled* in proportion to the model's current scores.
