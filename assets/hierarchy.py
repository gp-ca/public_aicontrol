import pandas as pd

# list of all celltypes except Q*
celltypes_all = ['BP', 'CD4', 'CD4x38', 'CD4xDR', 'CD4xDRx38', 'CD8', 'CD8x38', 'CD8xDR', 'CD8xDRx38', 'Lympho', 'NKP',
                 'NKTP', 'NKx38', 'NKxDR', 'NKxDRx38', 'T48NT', 'T48PT', 'T4P', 'T8P', 'TP']

# list of celltypes considered by the DDMs
celltypes_ddm = ['Lympho', 'T4P', 'T8P', 'BP', 'TP', 'NKP']

# list of celltypes considered by the expert
celltypes_immu = ['BP', 'CD4x38', 'CD4xDR', 'CD4xDRx38', 'CD8x38', 'CD8xDR', 'CD8xDRx38', 'Lympho', 'NKP',
                 'NKTP', 'NKx38', 'NKxDR', 'NKxDRx38', 'T48NT', 'T48PT', 'T4P', 'T8P', 'TP']

# columns names of celltypes considered by the DDMs
predicted_celltype_names = [f'pred_{cell}' for cell in celltypes_ddm]


def get_hierarchy_lookup():
    hierarchy_lookup = pd.DataFrame(
        index=['BP', 'TP', 'T48PT', 'T48NT', 'NKP', 'NKTP'])  # add before, otherwise key error

    # Lympho might have FSC-A or APC-H7-A as marker on the x-axis
    hierarchy_lookup.loc['Lympho', ['parent', 'marker_x', 'marker_y']] = 'all_events', 'FSC-A', 'SSC-A'
    hierarchy_lookup.loc['Lympho_alternative', ['parent', 'marker_x', 'marker_y']] = 'all_events', 'APC-H7-A', 'SSC-A'

    hierarchy_lookup.loc[['BP', 'TP'], ['parent', 'marker_x', 'marker_y']] = 'Lympho', 'AmCyan-A', 'APC-A'

    hierarchy_lookup.loc[['NKP', 'NKTP'], ['parent', 'marker_x', 'marker_y']] = 'Lympho', 'AmCyan-A', 'PE-Cy7-A'
    hierarchy_lookup.loc['T8P', 'parent'] = 'Lympho'
    hierarchy_lookup.loc['T4P', 'parent'] = 'Lympho'

    hierarchy_lookup.loc['CD4', ['parent', 'marker_x',
                                 'marker_y']] = 'Lympho', 'AmCyan-A', 'Pacific Blue-A'  # not on right table of PDF file
    hierarchy_lookup.loc[
        'CD8', ['parent', 'marker_x', 'marker_y']] = 'Lympho', 'AmCyan-A', 'PerCP-A'  # not on right table of PDF file

    hierarchy_lookup.loc[['T48PT', 'T48NT'], ['parent', 'marker_x', 'marker_y']] = 'TP', 'Pacific Blue-A', 'PerCP-A'

    hierarchy_lookup.loc['NKxDRx38', ['parent', 'marker_x', 'marker_y']] = 'NKP', 'PE-A', 'FITC-A'
    hierarchy_lookup.loc['NKxDR', 'parent'] = 'NKP'
    hierarchy_lookup.loc['NKx38', 'parent'] = 'NKP'

    hierarchy_lookup.loc['CD4xDRx38', ['parent', 'marker_x', 'marker_y']] = 'T4P', 'FITC-A', 'PE-A'
    hierarchy_lookup.loc['CD4x38', 'parent'] = 'T4P'
    hierarchy_lookup.loc['CD4xDR', 'parent'] = 'T4P'

    hierarchy_lookup.loc['CD8xDRx38', ['parent', 'marker_x', 'marker_y']] = 'T8P', 'FITC-A', 'PE-A'
    hierarchy_lookup.loc['CD8x38', 'parent'] = 'T8P'
    hierarchy_lookup.loc['CD8xDR', 'parent'] = 'T8P'

    # add celltypes per quadrant
    # q1 q2
    # q3 q4
    hierarchy_lookup.loc[['BP', 'TP'], ['q1', 'q2', 'q3', 'q4']] = 'BP', 'Q2', 'Q3', 'TP'
    hierarchy_lookup.loc['CD8', ['q1', 'q2', 'q3', 'q4']] = 'Q1-1', 'CD8', 'Q3-1', 'Q4-1'
    hierarchy_lookup.loc['CD4', ['q1', 'q2', 'q3', 'q4']] = 'Q1-2', 'CD4', 'Q3-2', 'Q4-2'
    hierarchy_lookup.loc[['T48PT', 'T48NT'], ['q1', 'q2', 'q3', 'q4']] = 'Q1-3', 'T48PT', 'T48NT', 'Q4-3'
    hierarchy_lookup.loc[['NKP', 'NKTP'], ['q1', 'q2', 'q3', 'q4']] = 'NKP', 'NKTP', 'Q3-4', 'Q4-4'
    hierarchy_lookup.loc['CD4xDRx38', ['q1', 'q2', 'q3', 'q4']] = 'Q1-6', 'CD4xDRx38', 'Q3-6', 'Q4-6'
    hierarchy_lookup.loc['CD8xDRx38', ['q1', 'q2', 'q3', 'q4']] = 'Q1-5', 'CD8xDRx38', 'Q3-5', 'Q4-5'
    hierarchy_lookup.loc['NKxDRx38', ['q1', 'q2', 'q3', 'q4']] = 'Q1-7', 'NKxDRx38', 'Q3-7', 'Q4-7'

    return hierarchy_lookup
