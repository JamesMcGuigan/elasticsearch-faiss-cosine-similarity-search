import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.alignment.aligner import Alignment


def visualize_similarities(
        similarities: np.ndarray,
        xlabel: str,
        ylabel: str,
        title='Cosine Similarities',
        filename: str = None,
        show    = True,
        verbose = False
) -> Union[str, None]:

    # plt.matshow(similarities)
    # plt.colorbar()
    sns.heatmap(
        data=similarities,  vmax=1, vmin=0, annot=True,
        xticklabels=range(similarities.shape[0]),
        yticklabels=range(similarities.shape[1]),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.gca().xaxis.set_label_position('top')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)  # call .savefig() before .show()
        if verbose: print(filename)
    if show:
        plt.show()
    plt.close()
    return filename


def _index_alignment_to_xy_values(alignment: Alignment):
    x_values, y_values = [0], [0]
    prev_source_index, prev_target_index = 0, 0
    for i, _ in enumerate(alignment):
        source_index, target_index = alignment[i]
        if source_index is None:
            x_values.append(target_index + 1)
            y_values.append(prev_source_index)
            prev_target_index = target_index + 1
        elif target_index is None:
            x_values.append(prev_target_index)
            y_values.append(source_index + 1)
            prev_source_index = source_index + 1
        else:
            x_values.append(target_index + 1)
            y_values.append(source_index + 1)
            prev_source_index = source_index + 1
            prev_target_index = target_index + 1

    return np.array(x_values) + 0.5, np.array(y_values) + 0.5


def _ladder_alignment_to_xy_values(alignment: Alignment):
    x_values, y_values = [], []
    for rung in alignment:
        source_sentence_boundary_index, target_sentence_boundary_index = rung
        x_values.append(target_sentence_boundary_index)
        y_values.append(source_sentence_boundary_index)

    return np.array(x_values), np.array(y_values)


def _alignment_to_xy_values(alignment: Alignment, alignment_type: str):
    if alignment_type == 'index':
        return _index_alignment_to_xy_values(alignment)
    if alignment_type == 'ladder':
        return _ladder_alignment_to_xy_values(alignment)
    
    raise ValueError(f'Unknown alignment type: \'{alignment_type}\'')


def visualize_alignment(
    similarities: np.ndarray, 
    candidate_alignment: Alignment,
    ground_truth_alignment: Alignment=None,
    alignment_type: str='index',
    xlabel: str = 'Target sentences',
    ylabel: str = 'Source sentences',
    title:  str ='',
):
    """Visualizes an alignment as a path along a similarity matrix.

    Args:
        similarities (np.ndarray): 2D NumPy array containing similarity
          scores between each pair of sentences.
        candidate_alignment (Alignment): The candidate alignment.
        ground_truth_alignment (Alignment, optional): The ground truth 
          alignment. Defaults to None.
    """

    # Set an appropriate figure size
    n_source_sentences, n_target_sentences = similarities.shape
    fig_height = 16
    # fig_width = int((n_target_sentences / n_source_sentences) * fig_height)
    # plt.figure(figsize=(fig_width, fig_height))
    plt.figure()  # defaults work better for small plots
    # Create a canvas to draw on
    border = int(alignment_type == 'index')
    canvas = np.zeros((
        n_source_sentences + border, 
        n_target_sentences + border
    ))
    canvas[border:, border:] = similarities
    # Plot the candidate_alignment on top of the canvas
    legenvd = ['Candidate alignment']
    ax = sns.heatmap(
        canvas,
        cmap='Greys',
        cbar=True,
        vmax=1, vmin=0, annot=True
    )
    x_values_cand, y_values_cand = _alignment_to_xy_values(
        candidate_alignment,
        alignment_type
    )
    for x, y in zip(x_values_cand, y_values_cand):
        ax.add_artist(plt.Circle((x, y), 0.1, color='red'))
    ax.plot(x_values_cand, y_values_cand, 'r-', linewidth=5)
    if ground_truth_alignment is not None:
        # Plot the ground truth alignment on top of similarity  matrix
        legenvd.append('Ground truth alignment')
        x_values_true, y_values_true = _alignment_to_xy_values(
            ground_truth_alignment,
            alignment_type
        )
        for x, y in zip(x_values_true, y_values_true):
            ax.add_artist(plt.Circle((x, y), 0.1, color='green'))
        ax.plot(x_values_true, y_values_true, 'g-', linewidth=5)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legenvd)

    plt.show()
