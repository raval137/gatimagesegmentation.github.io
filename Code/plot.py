import matplotlib.pyplot as plt

path = "/home/mpatel74/projects/def-akilan/mpatel74/TRY500/GAT2/try9/loss.txt"

train_loss = []
train_f1_score = []
val_loss = []
val_f1_score = []
temp = []
with open(path, "r") as f:
    for line in f.readlines():
        temp = line.split()
        train_loss.append(float(temp[4]))
        train_f1_score.append(float(temp[6]))
        val_loss.append(float(temp[9]))
        val_f1_score.append(float(temp[11]))


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
y1 = ax1.plot(range(1, len(train_loss)+1), train_loss,
              color="green", label="train_loss")

y2 = ax2.plot(range(1, len(train_f1_score)+1), train_f1_score,
              color="red", label="train_F1-score")

y3 = ax1.plot(range(1, len(val_loss)+1), val_loss,
              color="blue", label="val_loss")

y4 = ax2.plot(range(1, len(val_f1_score)+1),
              val_f1_score, color="orange", label="val_F1-score")
ax1.set_title(f"Plot for data aug")
ax1.set_xlabel('Epoch')
ax1.set_ylabel("Loss")
ax2.set_ylabel("F1-score")
leg = y1 + y2 + y3 + y4
labs = [l.get_label() for l in leg]
ax1.legend(leg, labs, loc=0)
plt.savefig(
    f"/home/mpatel74/projects/def-akilan/mpatel74/TRY500/GAT2/try9/try9.png", dpi=1000)
plt.show()
