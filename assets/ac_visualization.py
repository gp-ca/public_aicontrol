import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import numpy as np


def cell_network_graph_dqa(node_data, color_dict):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges (relationships)
    G.add_edge('all events', 'Lympho')
    G.add_edge('Lympho', 'BP')
    G.add_edge('Lympho', 'NKP')
    G.add_edge('Lympho', 'TP')
    G.add_edge('BP', 'NKP')
    G.add_edge('NKP', 'BP')
    G.add_edge('NKP', 'TP')
    G.add_edge('TP', 'NKP')  # Add reverse edge for bidirectional arrow
    G.add_edge('NKP', 'T4P')
    G.add_edge('T4P', 'NKP')  # Add reverse edge for bidirectional arrow
    G.add_edge('NKP', 'T8P')
    G.add_edge('T8P', 'NKP')  # Add reverse edge for bidirectional arrow
    G.add_edge('BP', 'NKP')  # Add reverse edge for bidirectional arrow
    G.add_edge('TP', 'T4P')
    G.add_edge('T4P', 'TP')  # Add reverse edge for bidirectional arrow
    G.add_edge('TP', 'T8P')
    G.add_edge('T8P', 'TP')  # Add reverse edge for bidirectional arrow
    G.add_edge('TP', 'NKTP')
    G.add_edge('T4P', 'NKT4P')
    G.add_edge('T8P', 'NKT8P')
    G.add_edge('NKP', 'BPNKP')
    G.add_edge('BP', 'BPNKP')

    # node data
    node_data = node_data

    # Normalize node
    node_sizes = [size / 250 for size in node_data.values()]

    # Specify positions
    pos = {'all events': (0, 5), 'Lympho': (3, 4.5), 'BP': (0.5, 3),
           'NKP': (2, 3), 'T8P': (3, 2.5), 'TP': (4, 3), 'T4P': (3, 3.5),
           'NKTP': (3, 3), 'NKT4P': (2.5, 3.25), 'NKT8P': (2.5, 2.75),
           'BPNKP': (1.25, 3)}

    # edge colors
    edge_colors = ['red' if 'BP' in edge and 'NKP' in edge else "gray" for
                   edge in G.edges()]

    labels = {'all events': 'all events',
              'Lympho': 'Lympho',
              'BP': 'BP',
              'NKP': 'NKP',
              'TP': 'TP',
              'T4P': 'T4P',
              'T8P': 'T8P',
              'NKTP': '!',
              'NKT4P': '!',
              'NKT8P': '!',
              'BPNKP': '!'}

    node_colors = color_dict.values()

    text_colors = {key: 'white' if key in ['all events', 'Lympho', 'BP']
                   else 'black' for key in labels.keys()}

    fig, ax = plt.subplots(figsize=(8.27/2*1.5, 11.69/3*1.5))

    # Draw networkx graph (nodes and edges)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.95)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=False)
    special_label = 'BPNKP'
    nx.draw_networkx_labels(G, pos,
                            labels={special_label: labels[special_label]},
                            font_color='black')
    # Draw labels with specified text colors
    for node, text_color in text_colors.items():
        nx.draw_networkx_labels(G, pos, labels={node: labels[node]},
                                font_color=text_color)

    ax.margins(0.15)
    plt.axis('off')
    plt.show()


def hist_standard_ranges(summed_expert1, cell_subsets_of_interst,
                         parents_cell_subset_of_interest, reported_ranges,
                         color_dict):
    fig, axs = plt.subplots(3, 2, figsize=(8.27/2*1.5, 11.69/3*1.5))
    sns.set_theme(style='whitegrid')

    handles, labels = [], []

    for i, (celltype, parent) in enumerate(
        zip(cell_subsets_of_interst, parents_cell_subset_of_interest)
    ):
        ratio = (summed_expert1.loc['051':][cell_subsets_of_interst] /
                 np.asarray(
                     summed_expert1.loc['051':]
                     [parents_cell_subset_of_interest]
                ))[[celltype]] * 100

        # Select subplot
        ax = axs[i//2, i % 2]
        # Create histogram
        counts, bins, patches = ax.hist(ratio, bins=12, alpha=0.7,
                                        color=color_dict[celltype],
                                        label='Histogram')

        # Calculate area of histogram
        area_hist = sum(np.diff(bins)*counts)

        # Create KDE plot
        kde = sns.kdeplot(data=ratio.squeeze(), color='b', ax=ax, label="KDE")

        # Get KDE line
        kde_line = kde.get_lines()[-1]
        x_kde, y_kde = kde_line.get_data()

        # Scale KDE line
        kde_line.set_ydata(y_kde*area_hist)

        # Get boundaries
        lower_boundary = reported_ranges.loc[celltype][0]
        upper_boundary = reported_ranges.loc[celltype][1]

        # Draw boundaries
        ax.axvline(lower_boundary, color='r', linestyle='--',
                   label='Standard range')
        ax.axvline(upper_boundary, color='r', linestyle='--')

        if i % 2 == 0:
            ax.set_ylabel('Number of donors')
        else:
            ax.set_ylabel('')
        # ax.set_title(f'{celltype}')
        ax.text(0.02, 0.98, celltype, transform=ax.transAxes,
                verticalalignment='top')
        ax.set_ylim([0, 12])

        if i == 4 or i == 5:
            ax.set_xlabel('CR expert 1 [%]')
        else:
            ax.set_xlabel('')

        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    # Remove duplicates
    handles, labels = list(dict(zip(labels, handles)).values()), list(dict(zip(labels, handles)).keys())

    fig.legend(handles, labels, loc='upper right', facecolor='white',
               framealpha=.8)

    plt.tight_layout()
    plt.show()


def plot_final_model_f1_scores(f1_scores_model, color_dict, save: bool,
                               loc: str):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8.27/2*1.5, 11.69/3*1.5/2))

    # Convert to a long-form
    long_df = f1_scores_model.melt(var_name='Class', value_name='F1 Score')

    # Assign colors based dictionary
    palette = [color_dict.get(x, (1, 1, 1)) for x in long_df['Class'].unique()]
    # Plot boxplot horizontally
    sns.stripplot(y='Class', x='F1 Score', data=long_df, palette=palette,
                  size=3, jitter=.09, alpha=0.8)
    sns.boxplot(y='Class', x='F1 Score', data=long_df,
                boxprops=dict(alpha=.3),
                showcaps=True,
                whiskerprops={'linewidth': 1.5, 'alpha': 0.5},
                showfliers=False,
                palette=palette)

    plt.ylabel('')
    plt.xlabel('F1 score')
    plt.xlim(0.825, 1) 

    sns.despine(offset=10, trim=True)
    plt.tight_layout()

    patches = []
    # Get unique classes
    classes = long_df['Class'].unique()
    for i, class_name in enumerate(classes):
        patches.append(mpatches.Patch(color=palette[i], label=class_name))
    # Add legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
               borderaxespad=0.)
    if save is True:
        plt.savefig(loc, dpi=300, bbox_inches='tight')
    plt.show()


def cell_network_graph_mqa(node_data, color_dict):
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges (relationships)
    G.add_edge('all events', 'Lympho')
    G.add_edge('Lympho', 'BP')
    G.add_edge('Lympho', 'NKP')
    G.add_edge('Lympho', 'TP')
    G.add_edge('NKP', 'TP')
    G.add_edge('TP', 'NKP')  # Add reverse edge for bidirectional arrow
    G.add_edge('NKP', 'T4P')
    G.add_edge('T4P', 'NKP')  # Add reverse edge for bidirectional arrow
    G.add_edge('TP', 'T4P')
    G.add_edge('T4P', 'TP')  # Add reverse edge for bidirectional arrow
    G.add_edge('TP', 'T8P')
    G.add_edge('T8P', 'TP')  # Add reverse edge for bidirectional arrow
    G.add_edge('TP', 'NKTP')
    G.add_edge('T4P', 'NKT4P')

    # node data
    node_data = node_data

    # Normalize node
    node_sizes = [size / 250 for size in node_data.values()]

    # Specify positions
    pos = {'all events': (0, 5), 'Lympho': (3, 4.5), 'BP': (0.5, 3),
           'NKP': (2, 3), 'T8P': (3, 2.5), 'TP': (4, 3), 'T4P': (3, 3.5),
           'NKTP': (3, 3), 'NKT4P': (2.5, 3.25)}

    # edge colors
    edge_colors = ['red' if 'BP' in edge and 'NKP' in edge else "gray" for
                   edge in G.edges()]

    labels = {'all events': 'all events',
              'Lympho': 'Lympho',
              'BP': 'BP',
              'NKP': 'NKP',
              'TP': 'TP',
              'T4P': 'T4P',
              'T8P': 'T8P',
              'NKTP': '!',
              'NKT4P': '!'}

    node_colors = color_dict.values()

    text_colors = {key: 'white' if key in ['all events', 'Lympho', 'BP']
                   else 'black' for key in labels.keys()}

    fig, ax = plt.subplots(figsize=(8.27/2*1.5, 11.69/3*1.5))

    # Draw networkx graph (nodes and edges)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.95)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=False)
    # Draw labels with specified text colors
    for node, text_color in text_colors.items():
        nx.draw_networkx_labels(G, pos, labels={node: labels[node]},
                                font_color=text_color)

    ax.margins(0.15)
    plt.axis('off')
    plt.show()
