import matplotlib.pyplot as plt

fig7, f7_axs = plt.subplots(ncols=2, nrows=5)
gs = f7_axs[-1, 0].get_gridspec()
# remove the underlying axes
for ax in f7_axs[-1, :]:
    ax.remove()
axbig = fig7.add_subplot(gs[-1, :])
axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
               xycoords='axes fraction', va='center')

fig7.tight_layout()

plt.show()
