from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from matplotlib.collections import PatchCollection
import matplotlib.lines as lines
import itertools
import argparse

#### TODO : faire en sorte que si il y a des labels dead/living la légende soit pas en italique #####

# Configuration du parser
parser = argparse.ArgumentParser(description="Génère une matrice de confusion avec ou sans la classe 'unsure'.")
parser.add_argument(
    "--wo_unsure",
    action="store_false",  # Si présent, la variable sera False, sinon True
    help="Exclure la classe 'unsure' dans les calculs et les graphiques."
)
args = parser.parse_args()

# Utilisation de l'argument
unsure = args.wo_unsure  # True si --unsure est passé, False sinon

df = pd.read_csv('test_labels_w_pred.csv')
unsure_preds = df[df['pred_label']=='unsure']
print('Pictures with unsure predictions:')
print(unsure_preds)
percentage_unsure_labels = round((unsure_preds.shape[0]/df.shape[0]) * 100, 2)
print(f'percentage of unsure labels: {percentage_unsure_labels}')

df_wo_unsure = df[~(df['pred_label'] == 'unsure')]
misidentif_preds = df_wo_unsure[df_wo_unsure['pred_label'] != df_wo_unsure['true_label']]

print('Pictures with wrong predictions:')
print(misidentif_preds)
percentage_wrong_labels = round((misidentif_preds.shape[0]/df.shape[0]) * 100, 2)
print(f'percentage of wrong labels: {percentage_wrong_labels}')

# Filtrer les données sans "unsure"
if not unsure:
    df = df[~(df['pred_label'] == 'unsure') & ~(df['true_label'] == 'unsure')]
    
y_true = df['true_label']
y_pred = df['pred_label']

cls_labels = sorted(set(y_true) | set(y_pred))

def get_text_width(txt):
    f = plt.figure()
    r = f.canvas.get_renderer()
    t = plt.text(0.5, 0.5, txt)
    plt.tight_layout()
    bb = t.get_window_extent(renderer=r)
    width = bb.width
    plt.close('all')
    return width

def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
def split_genus_species(label):
    """
    Convertit 'Genus_species' en 'Genus species' avec mise en italique
    via mathtext (Matplotlib LaTeX-like).
    """
    if '_' in label:
        genus, species = label.split('_', 1)
        latin = f"{genus} {species}"
        
        # convertir les espaces en "\ " pour mathtext
        latin_tex = latin.replace(" ", r"\ ")
        
        # renvoyer en italique
        return rf"$\it{{{latin_tex}}}$"

    return label  # 'unsure' ou labels non biologiques

def format_cls_labels(label, occurences):
    if label != 'unsure':
        return f"{split_genus_species(label)} ({occurences})"
    return "unsure"


def plot_confusion_accuracy_matrix(y_true, y_pred, cls_labels, normalise=True, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None, show=False):
    mult = 3
    max_word = cls_labels[np.argmax([len(f"{lab}") for lab in cls_labels])]
    txt_width = get_text_width(max_word) / 40
    sz = len(cls_labels) / mult + txt_width + 1.5
    sz2 = len(cls_labels) / mult + 1.5
    if figsize is None:
        if unsure:
            figsize = (sz, sz - 1)
        else:
            figsize = (sz, sz)

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)    
    cm = confusion_matrix(y_true, y_pred)
    
    # Retirer la ligne "unsure" (true_label) mais garder la colonne
    if 'unsure' in cls_labels:
        unsure_idx = cls_labels.index('unsure')

        # retirer la ligne dans la matrice
        cm = np.delete(cm, unsure_idx, axis=0)

        # retirer la classe dans les labels true (axe Y)
        cls_labels_y = [l for l in cls_labels if l != 'unsure']
    else:
        cls_labels_y = cls_labels

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm.astype('float')), where=(row_sums != 0))
        cm = np.round(cm * 100).astype(int)

    f, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [sz2 - 1.5, 1], 'height_ratios': [1, sz2 - 1.5], 'wspace': 0, 'hspace': 0}, figsize=figsize)
    cls_labels_with_counts = [format_cls_labels(cls_labels[i], s[i]) for i in range(len(cls_labels))]
    cls_labels_y_with_counts = [format_cls_labels(cls_labels_y[i], s[i]) for i in range(len(cls_labels_y))]
    thresh = cm.max() / 2.

    ax_cm = ax[1, 0]
    ax_right = ax[1, 1]
    ax_top = ax[0, 0]
    ax_unused = ax[0, 1]
    ax_unused.axis('off')
    
    tick_marks_x = np.arange(len(cls_labels))     
    tick_marks_y = np.arange(len(cls_labels_y))
    
    ax_cm.set_xticks(tick_marks_x)
    ax_cm.set_xticklabels(cls_labels_with_counts, rotation=90)
    ax_cm.set_yticks(tick_marks_y)
    ax_cm.set_yticklabels(cls_labels_y_with_counts)
    ax_cm.set_xlim(-0.5, len(cls_labels) - 0.5)
    ax_cm.set_ylim(-0.5, cm.shape[0] - 0.5)
    ax_cm.invert_yaxis()

    # Barres de precision et recall (sans "unsure")
    if unsure:
        remove_frame(ax_top)
        ax_top.bar(tick_marks_x[:-1], p[:-1], width=0.8, color=cmap(p), edgecolor=(0, 0, 0, 0.6))
        ax_top.set_xlim((-0.5, len(tick_marks_x) - 0.5))
        for i, v in enumerate(p[:-1]):
            clr = 'white' if np.mean(cmap(v)[:-1]) < 0.5 else 'black'
            ax_top.text(i, 0.15, '{:.1f}'.format(v * 100), color=clr, ha='center', rotation=90, alpha=0.7)

        remove_frame(ax_right)
        ax_right.barh(tick_marks_y, r[:-1], height=0.8, color=cmap(r), edgecolor=(0, 0, 0, 0.6))
        ax_right.set_ylim((-0.5, len(tick_marks_y) - 0.5))
        for i, v in enumerate(r[:-1]):
            clr = 'white' if np.mean(cmap(v)[:-1]) < 0.5 else 'black'
            ax_right.text(0.05, i, '{:.1f}'.format(v * 100), color=clr, va='center', alpha=0.7)
        ax_right.invert_yaxis()
    else:
        remove_frame(ax_top)
        ax_top.bar(tick_marks_x, p, width=0.8, color=cmap(p), edgecolor=(0, 0, 0, 0.6))
        ax_top.set_xlim((-0.5, len(tick_marks_x) - 0.5))
        for i, v in enumerate(p):
            clr = 'white' if np.mean(cmap(v)[:-1]) < 0.5 else 'black'
            ax_top.text(i, 0.15, '{:.1f}'.format(v * 100), color=clr, ha='center', rotation=90, alpha=0.7)

        remove_frame(ax_right)
        ax_right.barh(tick_marks_y, r, height=0.8, color=cmap(r), edgecolor=(0, 0, 0, 0.6))
        ax_right.set_ylim((-0.5, len(tick_marks_y) - 0.5))
        for i, v in enumerate(r):
            clr = 'white' if np.mean(cmap(v)[:-1]) < 0.5 else 'black'
            ax_right.text(0.05, i, '{:.1f}'.format(v * 100), color=clr, va='center', alpha=0.7)
        ax_right.invert_yaxis()

    # Matrice de confusion
    patches = []
    colors = []
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        patches.append(pch.Rectangle((j - 0.5, i - 0.5), 1, 1))
        colors.append(cm[i, j] / 100)
        ax_cm.text(j, i + 0.25, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    patcol = PatchCollection(patches, alpha=1, cmap=cmap)
    patcol.set_array(np.array(colors))
    ax_cm.add_collection(patcol)

    # Lignes de grille
    for i in range(len(cls_labels) - 1):  # On s'arrête à l'avant-dernière
        if i % 5 == 4:
            line1 = lines.Line2D([i + 0.5, i + 0.5], [-0.5, len(cls_labels) - 0.5], color=(0, 0, 0, 0.2))
            line2 = lines.Line2D([-0.5, len(cls_labels) - 0.5], [i + 0.5, i + 0.5], color=(0, 0, 0, 0.2))
            ax_cm.add_line(line1)
            ax_cm.add_line(line2)

    # Labels et titres
    ax_cm.set_ylabel('True label')
    ax_cm.set_xlabel('Predicted label')
    if unsure:
        ax_right.set_ylabel('Recall {:.1f}%'.format(np.mean(r[:-1]) * 100))
    else:
        ax_right.set_ylabel('Recall {:.1f}%'.format(np.mean(r) * 100))
    ax_right.yaxis.set_label_position('right')
    if unsure:
        ax_top.set_xlabel('Precision {:.1f}%'.format(np.mean(p[:-1]) * 100))
    else:
        ax_top.set_xlabel('Precision {:.1f}%'.format(np.mean(p) * 100))
    ax_top.xaxis.set_label_position('top')
    ax_top.set_title('Overall accuracy {:.1f}%'.format(accuracy_score(y_true, y_pred) * 100))
    ax_cm.set_zorder(100)
    plt.tight_layout()

    if normalise:
        print('saving normalized cm')
        f.patch.set_facecolor('white')
        plt.savefig("normalized_cm.pdf", facecolor='white', dpi=300, bbox_inches='tight')
    else:
        print('saving cm')
        f.patch.set_facecolor('white')
        plt.savefig("cm.pdf", facecolor='white', dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()

# Appel des fonctions
plot_confusion_accuracy_matrix(y_true, y_pred, cls_labels, normalise=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None, show=True)
plot_confusion_accuracy_matrix(y_true, y_pred, cls_labels, normalise=True, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None, show=True)
