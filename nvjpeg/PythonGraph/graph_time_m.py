# this files can create execution time graphs starting from the csvs created by the benchmarking program
import matplotlib.pyplot as plt
import csv

fmt = ["-gD", "-^", "-r*", "-ko"]
file_names = ["results_pinned_memm.csv", "results_mapped_memm.csv"]
n_imgs = 0
fig, ax = plt.subplots(2, 2, sharey=True)

for a in fig.axes:
    a.tick_params(
        axis='x',
        which='both',
        bottom=True,
        top=False,
        labelbottom=True
    )

for i, file in enumerate(file_names):
    with open(file, "r", newline="") as f:
        csvf = csv.DictReader(f)
        data = []
        for row in csvf:
            data.append(row)

    mem_type = data[0]["mem_type"] + " memory"
    n_imgs = data[0]["#imgs"]
    quality = list(dict.fromkeys(sorted([int(x["quality"]) for x in data])))
    subsampling = list(dict.fromkeys(sorted([x["subsampling"] for x in data])))
    optimizedData = [
        (x["subsampling"], int(x["quality"]), float(x["time"]))
        for x in data if int(x["optimizedHuffman"])
    ]
    optimizedData = sorted(optimizedData)
    nonoptimizedData = [
        (x["subsampling"], int(x["quality"]), float(x["time"]))
        for x in data if not int(x["optimizedHuffman"])
    ]
    nonoptimizedData = sorted(nonoptimizedData)

    title = ["Non-optimized Huffman ", "Optimized Huffman "]
    dataset = [nonoptimizedData, optimizedData]
    for k in range(2):
        for j, q in enumerate(quality):
            ax[k, i].title.set_text(title[k] + mem_type)
            ax[k, i].set_xlabel("chroma subsampling")
            ax[k, i].set_ylabel("time (ms)")
            ax[k, i].plot(
                subsampling,
                [x[2] for x in dataset[k] if x[1] == q],
                fmt[j],
                label=f"quality: {q}%"
            )
            ax[k, i].legend()

ax[0, 1].yaxis.set_tick_params(labelbottom=True)
ax[1, 1].yaxis.set_tick_params(labelbottom=True)
fig.suptitle(f"Time comparisons with a batch of {n_imgs} imgs")
plt.show()
