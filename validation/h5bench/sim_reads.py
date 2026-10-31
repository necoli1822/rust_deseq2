"""Simulate a transcriptome (genes with isoforms sharing exons) and paired-end reads.
usage: sim_reads.py <out dir> <n_genes> <n_samples> <reads per sample>"""
import os, random, sys

out, ng, ns, nr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
os.makedirs(out, exist_ok=True)
rnd = random.Random(7)
B = "ACGT"
comp = str.maketrans("ACGT", "TGCA")
tx = []  # (tx id, gene id, seq)
for g in range(ng):
    exons = ["".join(rnd.choice(B) for _ in range(rnd.randint(150, 600))) for _ in range(rnd.randint(2, 5))]
    for k in range(rnd.choice([1, 1, 2, 2, 3])):
        keep = [e for i, e in enumerate(exons) if i == 0 or k == 0 or rnd.random() < 0.7]
        tx.append((f"ENST{g:05d}.{k}", f"ENSG{g:05d}", "".join(keep)))
with open(f"{out}/tx.fa", "w") as f:
    for t, _, s in tx:
        f.write(f">{t}\n{s}\n")
with open(f"{out}/tx2gene.csv", "w") as f:
    f.write("TXNAME,GENEID\n")
    for t, g, _ in tx:
        f.write(f"{t},{g}\n")
base = [rnd.lognormvariate(0, 1.5) for _ in tx]
for j in range(ns):
    w = [b * rnd.lognormvariate(0, 0.4) * (2.0 if (j >= ns // 2 and i % 7 == 0) else 1.0) for i, b in enumerate(base)]
    tot = sum(wi * len(t[2]) for wi, t in zip(w, tx))
    cum, acc = [], 0.0
    for wi, t in zip(w, tx):
        acc += wi * len(t[2]) / tot
        cum.append(acc)
    import bisect
    with open(f"{out}/S{j+1}_1.fq", "w") as f1, open(f"{out}/S{j+1}_2.fq", "w") as f2:
        for r in range(nr):
            t = tx[min(bisect.bisect(cum, rnd.random()), len(tx) - 1)][2]
            frag = min(len(t), max(120, int(rnd.gauss(250, 40))))
            st = rnd.randint(0, len(t) - frag)
            fr = t[st:st + frag]
            a, b = fr[:75], fr[-75:].translate(comp)[::-1]
            q = "I" * 75
            f1.write(f"@r{r}/1\n{a}\n+\n{q[:len(a)]}\n")
            f2.write(f"@r{r}/2\n{b}\n+\n{q[:len(b)]}\n")
print(len(tx), "transcripts")
