import numpy as np
import fcsparser
import pandas as pd
from assets.hierarchy import get_hierarchy_lookup


# function to open .fcs file and convert data to numpy array
def open_fcs(fileName=""):
    # open fcs file
    header, data = fcsparser.parse(fileName, reformat_meta=True)
    # convert fcs data to numpy array
    fcs_data = np.array(data)
    return fcs_data


# define a function to automatically import an .fcs-file and compensate the
# data using spillover matrix
def importFCS_compensate(fileName="", spilloverMatrix=np.array):
    # spilloverMatrix should be given as NxN numpy array, with N = Number of
    # Fluorescence Channels

    # open fcs file
    header, data = fcsparser.parse(fileName, reformat_meta=True)

    # convert fcs data to numpy array
    fcs_data = np.array(data)

    # initialize spillover matrix (here: use the spillover matrix that was
    # manually defined before)
    spillover_mat = spilloverMatrix

    # compensate fcs_data
    compensation_mat = np.linalg.inv(spillover_mat.astype('f4'))
    fcs_data[:, 3:] = np.tensordot(fcs_data[:, 3:], compensation_mat,
                                   axes=([1], [0]))

    return header, fcs_data


# load channel names
named_channels = ['FSC-A', 'SSC-A', 'Anti-HLA-DR', 'CD38', 'CD8', 'CD16+56',
                  'CD19', 'CD45', 'CD4', 'CD3']
fluorochromes = ['FSC-A', 'SSC-A', 'FITC-A', 'PE-A', 'PerCP-A', 'PE-Cy7-A',
                 'APC-A', 'APC-H7-A', 'Pacific Blue-A', 'AmCyan-A']
cheat_sheet = []
# get antibodies from metadata
for i in range(10):
    cheat_sheet.append([i, named_channels[i], fluorochromes])
pd.DataFrame(cheat_sheet)

marker_dict = {
    'FSC-A': 'FSC-A',
    'SSC-A': 'SSC-A',
    'Anti-HLA-DR': 'FITC-A',
    'CD38': 'PE-A',
    'CD8': 'PerCP-A',
    'CD16+56': 'PE-Cy7-A',
    'CD19': 'APC-A',
    'CD45': 'APC-H7-A',
    'CD4': 'Pacific Blue-A',
    'CD3': 'AmCyan-A'
}


def preprocess_raw_data(fcs_data, labels, index):
    df1 = pd.concat({'measurement_data_compensated':
                     pd.DataFrame(fcs_data, columns=fluorochromes)},
                    axis=1, names=['', ''])

    df2 = pd.concat({'reported_labels': labels}, axis=1, names=['', ''])

    sample_information = pd.concat([df1, df2], axis=1)

    multi_index = pd.MultiIndex.from_tuples(
        list(zip(*(np.full(len(fcs_data), str(index+1).zfill(3)),
                   range(len(fcs_data))))),
        names=['file', 'event'])
    return sample_information.set_index(multi_index)


# define functions to generate joint frequency matrix
def determine_overlap_celltype_pair(df, celltype1, celltype2):
    return len(df[(df[celltype1] == 1) & (df[celltype2] == 1)])


def determine_overlap_celltypes_all(df, celltypes: list):
    matrix = pd.DataFrame(index=celltypes, columns=celltypes)
    for celltype1 in celltypes:
        for celltype2 in celltypes:
            if matrix.loc[celltype1, :].isna()[celltype2]:
                n_cells_overlap = determine_overlap_celltype_pair(
                    df, celltype1, celltype2
                    )
                matrix.loc[celltype1, celltype2] = n_cells_overlap
                matrix.loc[celltype2, celltype1] = n_cells_overlap
    return matrix


# function to compare cell subset ratios with standard ranges
def compare_to_standard_range(summed_events_df, celltype, parent,
                              lower_bound, upper_bound):
    true_value = summed_events_df.loc[:, celltype] / \
        summed_events_df.loc[:, parent]
    # number of events classified as the celltype is outside the reference
    # range
    return (true_value >= lower_bound) & (true_value <= upper_bound)


def cv_literature(df1, df2, celltypes, parents):
    # for detailes see Maecker et al. 2005 DOI:10.1186/1471-2172-6-13
    rel1 = np.asarray(df1[celltypes]) / np.asarray(df1[parents])
    rel2 = np.asarray(df2[celltypes]) / np.asarray(df2[parents])
    stack = np.stack((rel1, rel2))
    std = stack.std(axis=0)
    mean = stack.mean(axis=0)
    return 100*std/mean


# function to create figure for the section
def create_barplot(celltype, parent, summed_expert1, summed_expert2):
    summed_expert1_selected = summed_expert1.loc[:, celltype]
    summed_expert2_selected = summed_expert2.loc[:, celltype]
    # relative difference between summed events for this celltype, relative to
    # the total number of events
    fig_data = (summed_expert1_selected / summed_expert1.loc[:, parent] -
                summed_expert2_selected / summed_expert2.loc[:, parent])
    return fig_data


# function to create figure for the section
def create_scatter(celltype, parent, summed_expert1, summed_expert2):
    fig_data = pd.DataFrame()
    # todo patient ID -> sort
    fig_data.loc[:, 'Gating_expert1'] = summed_expert1.loc[:, celltype] / \
        summed_expert1.loc[:, parent]
    fig_data.loc[:, 'Gating_expert2'] = summed_expert2.loc[:, celltype] / \
        summed_expert2.loc[:, parent]
    return fig_data


def create_multi_label(df):
    unique_class_map = {'0-0-0-0-0-0': 'None',
                        '1-0-0-1-1-0': 'T4P',
                        '1-1-0-0-0-0': 'BP',
                        '1-0-1-0-0-0': 'NKP',
                        '1-0-0-1-0-1': 'T8P',
                        '1-0-0-1-0-0': 'TP',
                        '1-0-1-1-1-0': 'NKT4P',
                        '1-0-0-0-0-0': 'Lympho',
                        '1-1-1-0-0-0': 'BPNKP',
                        '1-0-1-1-0-0': 'NKTP',
                        '1-0-1-1-0-1': 'NKT8P'}
    df.loc[:, 'unique_class'] = df.astype(str).\
        agg('-'.join, axis=1).map(unique_class_map)
    return df['unique_class'].to_list()


def model_quality_assurance_preprocessing(d1, l, lt, prediction, index):
    df1 = pd.concat({'measurement_data_compensated':
                     d1},
                    axis=1, names=['', ''])
    df2 = pd.concat({'reported_labels': l}, axis=1, names=['', ''])
    df3 = pd.DataFrame(np.where((l['BP'] == 1) & (l['NKP'] == 1), 0, 1),
                       columns=['valid_Q_file'])
    df3 = pd.concat({'valid_Q': df3}, axis=1, names=['', ''])
    df4 = np.zeros(len(d1))
    df4[(lt,)] = 1
    df4 = pd.DataFrame(df4.astype(int), columns=['used_for_training_Q_file'])
    df4 = pd.concat({'used_for_training_Q': df4}, axis=1, names=['', ''])
    df5 = pd.concat({'predicted_labels': prediction}, axis=1, names=['', ''])
    sample_information = pd.concat([df1, df2, df3, df4, df5], axis=1)
    multi_index = pd.MultiIndex.from_tuples(
        list(zip(*(np.full(len(d1), str(index+1).zfill(3)), range(len(d1))))),
        names=['file', 'event'])
    return sample_information.set_index(multi_index)


def gate_for_celltype(patient_id, data_patient, celltype):
    hierarchy_lookup = get_hierarchy_lookup()
    # get parent celltype of current celltype
    parent_celltype = hierarchy_lookup.loc[celltype, 'parent']

    # consider only events of parent celltype
    data_gating = data_patient.loc[data_patient[parent_celltype] == 1]

    # celltypes of each quandrant
    q1 = hierarchy_lookup.loc[celltype, 'q1']
    q2 = hierarchy_lookup.loc[celltype, 'q2']
    q3 = hierarchy_lookup.loc[celltype, 'q3']
    q4 = hierarchy_lookup.loc[celltype, 'q4']

    # data points in each quadrant
    data_q1 = data_gating.loc[data_gating[q1] == 1]
    data_q2 = data_gating.loc[data_gating[q2] == 1]
    data_q3 = data_gating.loc[data_gating[q3] == 1]
    data_q4 = data_gating.loc[data_gating[q4] == 1]

    data_q1_q2 = data_gating.loc[(data_gating[q1] == 1) | (data_gating[q2] == 1)]
    data_q3_q4 = data_gating.loc[(data_gating[q3] == 1) | (data_gating[q4] == 1)]

    data_q1_q3 = data_gating.loc[(data_gating[q1] == 1) | (data_gating[q3] == 1)]
    data_q2_q4 = data_gating.loc[(data_gating[q2] == 1) | (data_gating[q4] == 1)]

    # determine horizontal gate

    marker_y = hierarchy_lookup.loc[celltype, 'marker_y']

    # determine if there is no split within the horizontal gate
    if data_q1_q2[marker_y].min() < data_q3_q4[marker_y].max():

        # horizontal gate for left side of gating plot
        min_q1 = data_q1[marker_y].min()
        max_q3 = data_q3[marker_y].max() 
        horizontal_gate_left = (min_q1 + max_q3) / 2.

        # horizontal gate for right side of gating plot
        min_q2 = data_q2[marker_y].min()
        max_q4 = data_q4[marker_y].max()
        horizontal_gate_right = (min_q2 + max_q4) / 2.

    else:
        # no split in horizontal gate
        min_q1_q2 = data_q1_q2[marker_y].min()
        max_q3_q4 = data_q3_q4[marker_y].max()

        horizontal_gate = (min_q1_q2 + max_q3_q4) / 2.
        horizontal_gate_left = horizontal_gate  # just one horizontal gate
        horizontal_gate_right = horizontal_gate  # just one horizontal gate

    # determine vertical gate

    marker_x = hierarchy_lookup.loc[celltype, 'marker_x']

    # determine if there is no split within the vertical gate
    if data_q2_q4[marker_x].min() < data_q1_q3[marker_x].max():

        # vertical gate for top part of gating plot
        min_q2 = data_q2[marker_x].min()
        max_q1 = data_q1[marker_x].max()
        vertical_gate_top = (min_q2 + max_q1) / 2.

        # vertical gate for bottom part of gating plot
        min_q4 = data_q4[marker_x].min()
        max_q3 = data_q3[marker_x].max()
        vertical_gate_bottom = (min_q4 + max_q3) / 2.

    else:
        # no split in vertical gate
        max_q1_q3 = data_q1_q3[marker_x].max()
        min_q2_q4 = data_q2_q4[marker_x].min()

        vertical_gate = (max_q1_q3 + min_q2_q4) / 2.
        vertical_gate_top = vertical_gate  # just one vertical gate
        vertical_gate_bottom = vertical_gate  # just one vertical gate

    return horizontal_gate_left, horizontal_gate_right, vertical_gate_top, vertical_gate_bottom


def gate_for_patient(patient_id, data):
    data_patient = data.loc[data['file']==str(patient_id).zfill(3), :]

    gating_patient_df = pd.DataFrame()
    hierarchy_lookup = get_hierarchy_lookup()
    # Names of celltypes determined in gating plots
    gating_celltype_names = list(hierarchy_lookup.loc[hierarchy_lookup['marker_x'].notna()].index.values)
    gating_celltype_names.remove('Lympho')
    gating_celltype_names.remove('Lympho_alternative')

    gating_celltype_names.remove('NKxDRx38')
    gating_celltype_names.remove('CD4xDRx38')
    gating_celltype_names.remove('CD8xDRx38')

    for celltype in gating_celltype_names:
        horizontal_gate_left, horizontal_gate_right, vertical_gate_top, vertical_gate_bottom = \
            gate_for_celltype(patient_id, data_patient, celltype)
        #print(horizontal_gate, vertical_gate)

        marker_x = hierarchy_lookup.loc[celltype, 'marker_x']
        marker_y = hierarchy_lookup.loc[celltype, 'marker_y']

        gating_patient_df.loc[patient_id, f'{celltype}_{marker_x}_q12'] = vertical_gate_top
        gating_patient_df.loc[patient_id, f'{celltype}_{marker_x}_q34'] = vertical_gate_bottom
        gating_patient_df.loc[patient_id, f'{celltype}_{marker_y}_q13'] = horizontal_gate_left
        gating_patient_df.loc[patient_id, f'{celltype}_{marker_y}_q24'] = horizontal_gate_right

    return gating_patient_df
