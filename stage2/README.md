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
