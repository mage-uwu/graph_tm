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
