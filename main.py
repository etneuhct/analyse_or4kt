import re
from collections import defaultdict, OrderedDict

import numpy
import pandas
from factor_analyzer import ConfirmatoryFactorAnalyzer, ModelSpecificationParser
from scipy.stats import spearmanr, chi2_contingency
from semopy import report

from cfa_modeling import compute_cfa_models
from utils import add_result_for_export, export_to_excel, all_subsets


def analytics(**kwargs):
    description = kwargs.get("description", "")
    ANALYTICS_RESUME.append({"description": description})


def import_data(data_path: str, separator: str = ";") -> pandas.DataFrame:
    data = pandas.read_csv(data_path, sep=separator)
    return data


def get_or4kt_variables(dataframe: pandas.DataFrame):
    all_columns = dataframe.columns
    regex = "d[0-7]_[0-9]+_(fr|en)"
    result = [element for element in all_columns if re.match(regex, element)]
    return result


def get_variables_in_different_language(dataframe: pandas.DataFrame):
    all_columns = dataframe.columns
    regex = ".*_(fr|en)"
    result = [element for element in all_columns if re.match(regex, element)]
    return result


def drop_or4kt_na_row(dataframe: pandas.DataFrame):
    variables = get_merged_or4kt_variables(dataframe)
    for variable in variables:
        dataframe[variable].replace({6: numpy.nan}, inplace=True)


def remove_empty_rows(dataframe: pandas.DataFrame):
    variables = get_or4kt_variables(dataframe)
    dataframe.dropna(axis=0, how="all", subset=variables, inplace=True)
    analytics(description="Les lignes contenant entierement la valeur 0 pour les variables de l'or4kt sont supprimées.")


def drop_or4kt_incoherent_rows(dataframe: pandas.DataFrame):
    new_variables = get_merged_or4kt_variables(dataframe)
    for variable in new_variables:
        dataframe.drop(dataframe[dataframe[variable] == 0].index, inplace=True)
    analytics(
        description="Les lignes contenant au moins une fois la valeur 0 pour une des"
                    " variables de l'or4kt fusionnées sont supprimées.")


def remove_second_arm(dataframe: pandas.DataFrame):
    dataframe.drop(dataframe[dataframe['redcap_event_name'] == "t2__collecte_arm_1"].index, inplace=True)
    analytics(description="Supprime les entrées correspondant au deuxième temps de mesure")


def fill_multiple_language_variables_na_with_0(dataframe: pandas.DataFrame):
    variables = get_variables_in_different_language(dataframe)
    for variable in variables:
        if dataframe[variable].dtype == 'object':
            dataframe[variable].fillna('', inplace=True)
        else:
            dataframe[variable].fillna(0, inplace=True)
    analytics(description="Remplace les valeurs N.A des variables divisées / langue par str vide ou 0")


def merge_variable_by_languages(dataframe: pandas.DataFrame):
    variables = get_variables_in_different_language(dataframe)
    new_variables = set("_".join(element.split('_')[:-1]) for element in variables)
    for new_variable in new_variables:
        dataframe.eval(f"{new_variable} = {new_variable}_fr + {new_variable}_en", inplace=True)
    analytics(description="Fusionne les variables divisées / langue")


def get_merged_or4kt_variables(dataframe: pandas.DataFrame):
    variables = get_or4kt_variables(dataframe)
    new_variables = set("_".join(element.split('_')[:2]) for element in variables)
    return list(new_variables)


def compute_correlation(dataframe: pandas.DataFrame):
    variables = sorted(get_merged_or4kt_variables(dataframe))
    series = dataframe[variables]
    correlation = series.corr(method="spearman")
    variable_groups = defaultdict(list)
    result = [{'data': correlation, "name": "correlation_global",
               "note": "d1_4 & d1_5 ont généralement des "
                       "correlations négatives avec les autres items. Le sens a été changé"}]
    for element in variables:
        variable_groups[element.split("_")[0]].append(element)
    for group_name in variable_groups:
        df = dataframe[variable_groups[group_name]]
        rho = df.corr()
        p_val = df.corr(method=lambda x, y: spearmanr(x, y)[1]) - numpy.eye(*rho.shape)
        p = p_val.applymap(lambda x: ''.join(['*' for t in [0.01, 0.05, 0.1] if x <= t]))
        corr_and_p_value = rho.round(2).astype(str) + p

        result.append(
            {"data": corr_and_p_value, "name": f"correlation_{group_name}_with_p"}
        )
    analytics(description="Calcul de corrélations.")

    return result


def compute_or4kt_description(dataframe: pandas.DataFrame):
    variables = sorted(get_merged_or4kt_variables(dataframe))
    description = dataframe[variables].describe()
    description.loc['% N.A'] = dataframe[variables].isnull().mean() * 100
    description = description.transpose()
    frequencies = compute_variables_frequencies(dataframe, variables)
    analytics(description="Calcul de fréquences & stats descriptives pour les var or4kt")
    return [
        {"data": description, "name": "or4kt description", 'note': 'R.A.S.'},
        {"data": frequencies, "name": "or4kt frequencies", 'note': 'R.A.S.'},
    ]


def compute_social_description(dataframe: pandas.DataFrame):
    description = dataframe[SOCIAL_VARIABLES].describe()
    description.loc['% N.A'] = dataframe[SOCIAL_VARIABLES].isnull().mean() * 100
    description = description.transpose()
    frequencies = compute_variables_frequencies(dataframe, SOCIAL_VARIABLES)
    analytics(description="Calcul de fréquences & stats descriptives pour le sociodémo")
    return [
        {"data": description, "name": "social_variables_description"},
        {"data": frequencies, "name": "social_variables_frequencies",
         "note": "Les variables 'duree_emploi', 'type_emploi', 'organisation' "
                 "sont libres et ne seront pas exploitables facilement."},
    ]


def drop_useless_variables(dataframe: pandas.DataFrame):
    dim_columns = [f'dim{i}_prime' for i in range(24)]
    columns = SOCIAL_VARIABLES + get_merged_or4kt_variables(dataframe) + dim_columns
    remove_variables = [element for element in dataframe.columns if element not in columns]
    dataframe.drop(columns=remove_variables, inplace=True)
    analytics(description="Suppression de variables qui ne sont ni or4kt ni socio demo")


def compute_variables_frequencies(dataframe: pandas.DataFrame, variables):
    counts = []
    frequencies = []
    modalities = []
    result_variables = []
    for key in variables:
        var_count = dataframe[key].value_counts(dropna=False).to_dict(OrderedDict)
        var_freq = dataframe[key].value_counts(normalize=True, dropna=False).to_dict(OrderedDict)
        counts = counts + list(var_count.values())
        frequencies = frequencies + list(var_freq.values())
        modalities = modalities + list(var_count.keys())
        result_variables = result_variables + [key for _ in range(len(var_count.keys()))]
    table = {
        "variables": result_variables,
        "modalities": modalities,
        "n": counts,
        "frequencies": frequencies
    }
    df = pandas.DataFrame(table)
    return df


def extract_incoherent_rows(dataframe: pandas.DataFrame):
    new_variables = get_merged_or4kt_variables(dataframe)
    indexes = []
    for variable in new_variables:
        index = dataframe[dataframe[variable] == 0].index
        indexes += index.to_list()
    filtered_dataframe = dataframe.filter(items=sorted(list(set(indexes))), axis=0)
    analytics(description="Extraits les lignes incohérentes")
    return [
        {
            "name": "incoherent_data",
            "data": filtered_dataframe,
            "note": "Ces entrées comportent des valeurs sans réponses pour des variables de l'or4. "
        }
    ]


def compute_cross_tables_for_or4kt(dataframe: pandas.DataFrame):
    cross_tabs = []
    for or4 in sorted(get_merged_or4kt_variables(dataframe)):
        if 6 in dataframe[or4].unique():
            for social_variable in ['language']:
                cross = pandas.crosstab([dataframe[social_variable]], dataframe[or4])
                modality_column = "variables"
                cross[modality_column] = cross.index
                cross.reset_index(drop=True, inplace=True)
                new_row = [f"{or4}/{social_variable}" if element == modality_column else '' for element in
                           cross.columns]
                cross.loc[-1] = new_row
                cross.index = cross.index + 1
                cross = cross.sort_index()
                cross_tabs.append(cross)

    df = pandas.concat(cross_tabs)
    analytics(description="Tableau croisé langage vs or4kt")

    return [
        {
            "name": "crosstab_or4kt_by_lg", "data": df,
            "note": "les effectifs sont en général insuffisant pour effectuer un chi² (< 5 / catégorie)."
        }
    ]


def compute_cross_table_na_by_language(dataframe: pandas.DataFrame):
    cross = pandas.crosstab([dataframe["language"]], dataframe["global_na"])
    c, p, dof, expected = chi2_contingency(cross)
    analytics(description="Calcul de chi² global_na vs language")
    return [
        {"name": "crosstab_na", "data": cross, "note": f"c: {c}, p_value: {p}"}
    ]


def export_clean_data(dataframe: pandas.DataFrame):
    en_data = dataframe.drop(dataframe[dataframe['language'] == 1].index)
    fr_data = dataframe.drop(dataframe[dataframe['language'] == 2].index)
    analytics(description="export clean data")
    return [
        {"name": "clean_data", "data": dataframe},
        {"name": "clean_data_en", "data": en_data},
        {"name": "clean_data_fr", "data": fr_data},
    ]


def export_cfa_variables(dataframe: pandas.DataFrame):
    df = dataframe.drop(columns=SOCIAL_VARIABLES)
    analytics(description="export cfa data")
    return [
        {"name": "cfa_data", "data": df},
    ]


def add_global_or4kt_na_variable(dataframe: pandas.DataFrame):
    indexes = []
    for variable in get_or4kt_variables(dataframe):
        index = dataframe[dataframe[variable] == 6].index.to_list()
        indexes = index + indexes
    indexes = sorted(list(set(indexes)))
    dataframe.eval(f"global_na = index in {indexes}", inplace=True)
    analytics(description="Crée une var 'global_na' True si au moins une variable or4kt = 6")


def change_variables_orientation(dataframe: pandas.DataFrame):
    variables = ["d1_4", "d1_5"]
    for variable in variables:
        dataframe[variable].replace({5: 1, 4: 2, 3: 3, 2: 4, 1: 5}, inplace=True)
    analytics(description="Changement du sens des var d1_4 & d1_5")


def compute_cfa(dataframe: pandas.DataFrame, model):
    model_spec = ModelSpecificationParser.parse_model_specification_from_dict(dataframe, model)
    cfa_analyzer = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    cfa_analyzer.fit(dataframe.values)
    factor_loading = pandas.DataFrame(data=cfa_analyzer.loadings_, columns=model_spec.factor_names, index=None)
    factor_loading["variables"] = model_spec.variable_names
    columns = ["variables"] + list(factor_loading.columns[:-1])
    factor_loading = factor_loading[columns]
    return factor_loading
    # return [{"name": "factor loadings (cfa)", "data": factor_loading}]


def compute_initial_cfa(dataframe: pandas.DataFrame):
    variables = sorted(get_merged_or4kt_variables(dataframe))
    model = defaultdict(list)
    for element in variables:
        model[element.split("_")[0]].append(element)
    new_db = dataframe[variables].dropna(how="all")
    factor_loadings = compute_cfa(new_db, model)
    return [{"name": "factor loadings (cfa)", "data": factor_loadings}]


def compute_sub_item_cfa(dataframe: pandas.DataFrame):
    model = {
        "dim1": ["d1_1", "d1_2", "d1_3"],
        "dim2": ["d1_4", "d1_5"],
        "dim3": ["d1_6", "d1_7", "d1_8"],
        "dim4": ["d1_9", "d1_10"],
        "dim5": ["d2_1"],
        "dim6": ["d2_2", "d2_3", "d2_4", "d2_5"],
        "dim7": ["d2_6", "d2_7", "d2_8", "d2_9", "d2_10"],
        "dim8": ["d3_1", "d3_2"],
        "dim9": ["d3_3", "d3_4", "d3_5"],
        "dim10": ["d3_6", "d3_7", "d3_8"],
        "dim11": ["d3_9"],
        "dim12": ["d4_1", "d4_2", "d4_3"],
        "dim13": ["d4_4"],
        "dim14": ["d4_5", "d4_6", "d4_7"],
        "dim15": ["d4_8", "d4_9", "d4_10"],
        "dim16": ["d5_1", "d5_2", "d5_3", "d5_4"],
        "dim17": ["d5_5", "d5_6"],
        "dim18": ["d5_7", "d5_8", "d5_9"],
        "dim19": ["d5_10"],
        "dim20": ["d6_1", "d6_2", "d6_3", "d6_4", "d6_5"],
        "dim21": ["d6_6"],
        "dim22": ["d6_7", "d6_8"],
        "dim23": ["d6_9", "d6_10"]
    }
    variables = sorted(get_merged_or4kt_variables(dataframe))
    new_db = dataframe[variables].dropna(how="all")
    factor_loadings = compute_cfa(new_db, model)
    return [{"name": "factor loadings (cfa)", "data": factor_loadings}]


def cfa_semopy(data):
    from semopy.inspector import inspect as sem_inspect
    from semopy import Model
    mod = """
    d1 =~ d1_1 + d1_10 + d1_2 + d1_3 + d1_4 + d1_5 + d1_6 + d1_7 + d1_8 + d1_9
    d2 =~ d2_1 + d2_10 + d2_2 + d2_3 + d2_4 + d2_5 + d2_6 + d2_7 + d2_8 + d2_9
    d3 =~ d3_1 + d3_2 + d3_3 + d3_4 + d3_5 + d3_6 + d3_7 + d3_8 + d3_9
    d4 =~ d4_1 + d4_10 + d4_2 + d4_3 + d4_4 + d4_5 + d4_6 + d4_7 + d4_8 + d4_9
    d5 =~ d5_1 + d5_10 + d5_2 + d5_3 + d5_4 + d5_5 + d5_6 + d5_7 + d5_8 + d5_9
    d6 =~ d6_1 + d6_10 + d6_2 + d6_3 + d6_4 + d6_5 + d6_6 + d6_7 + d6_8 + d6_9
    d1 ~~ d2
    d1 ~~ d3
    d1 ~~ d4
    d1 ~~ d5
    d1 ~~ d6
    """
    df = data[get_merged_or4kt_variables(data)]
    # df = df.dropna(how='any')
    model = Model(mod, mimic_lavaan=True)
    # model.load_dataset(df)
    model.fit(df)
    # opt = Optimizer(model)
    # o = opt.optimize()
    a = sem_inspect(model, std_est=True)
    print(a)
    from semopy import gather_statistics
    stats = gather_statistics(model)
    from semopy import semplot
    g = semplot(model, "pd.png")
    report(model, "Political Democracy")


def compute_sub_dim(data):
    structure = SUB_DIM_STRUCTURE
    for new_variable in structure:
        data.eval(f'{new_variable}_prime = ({" + ".join(structure[new_variable])}) / {len(structure[new_variable])}',
                  inplace=True)


def compute_cronbach(data):
    model = SUB_DIM_STRUCTURE
    result = []
    for key in model:
        items = data[model[key]]
        if items.shape[1] < 2:
            continue
        items = items.dropna(how='any')
        score = cronbach_alpha_by_factor(items)
        shape = items.shape
        obs_nb = shape[0]
        items_optimized, score_optimized, obs_optimized = cronbach_optimisation(data, model[key])
        result.append({
            "factors": key,
            "variables": ', '.join(model[key]),
            "cronbach_score": score,
            "n": obs_nb,
            "items_optimized": items_optimized,
            "score_optimized": score_optimized,
            "n_optimized": obs_optimized
        })
    result_df = pandas.DataFrame.from_records(result)
    return [{"data": result_df, "name": "cronbach"}]


def cronbach_optimisation(database, variables):
    minimum = len(variables) - 1 if len(variables) > 2 else 2
    subsets = all_subsets(variables, minimum=minimum)
    best_score = 0
    best_shape = None
    best_subset = None
    for subset in subsets:
        items = database[list(subset)].dropna(how="any")
        score = cronbach_alpha_by_factor(items)
        if abs(best_score) < abs(score):
            best_score = score
            best_subset = subset
            best_shape = items.shape[0]
    return ', '.join(best_subset), best_score, best_shape


def cronbach_alpha_by_factor(items) -> float:
    items_count = items.shape[1]
    variance_sum = float(items.var(axis=0, ddof=1).sum())
    total_var = float(items.sum(axis=1).var(ddof=1))
    return items_count / float(items_count - 1) * (1 - variance_sum / total_var)


def main():
    data = import_data("data/data.csv")
    result_to_export = []

    # cleaning 1
    remove_empty_rows(data)
    fill_multiple_language_variables_na_with_0(data)
    add_global_or4kt_na_variable(data)
    merge_variable_by_languages(data)

    # compute
    incoherent_rows = extract_incoherent_rows(data)
    cross_table_na_by_language = compute_cross_table_na_by_language(data)

    # cleaning
    remove_second_arm(data)
    drop_or4kt_incoherent_rows(data)

    #
    cross_tab_for_or4kt = compute_cross_tables_for_or4kt(data)

    drop_or4kt_na_row(data)
    change_variables_orientation(data)
    social_description = compute_social_description(data)
    compute_sub_dim(data)

    # computing
    correlations = compute_correlation(data)
    or4kt_description = compute_or4kt_description(data)
    compute_cfa_models(data)
    cronbach = compute_cronbach(data)
    #
    drop_useless_variables(data)

    #
    clean_datas = export_clean_data(data)
    cfa_variables = export_cfa_variables(data)

    # add for export
    add_result_for_export(result_to_export, correlations)
    add_result_for_export(result_to_export, or4kt_description)
    add_result_for_export(result_to_export, social_description)
    add_result_for_export(result_to_export, clean_datas)
    add_result_for_export(result_to_export, incoherent_rows)
    add_result_for_export(result_to_export, cross_tab_for_or4kt)
    add_result_for_export(result_to_export, cross_table_na_by_language)
    add_result_for_export(result_to_export, cfa_variables)
    add_result_for_export(result_to_export, cronbach)

    export_to_excel(result_to_export)


if __name__ == '__main__':
    SUB_DIM_STRUCTURE = {
        "dim1": ["d1_1", "d1_2", "d1_3"],
        "dim2": ["d1_4", "d1_5"],
        "dim3": ["d1_6", "d1_7", "d1_8"],
        "dim4": ["d1_9", "d1_10"],
        "dim5": ["d2_1"],
        "dim6": ["d2_2", "d2_3", "d2_4", "d2_5"],
        "dim7": ["d2_6", "d2_7", "d2_8", "d2_9", "d2_10"],
        "dim8": ["d3_1", "d3_2"],
        "dim9": ["d3_3", "d3_4", "d3_5"],
        "dim10": ["d3_6", "d3_7", "d3_8"],
        "dim11": ["d3_9"],
        "dim12": ["d4_1", "d4_2", "d4_3"],
        "dim13": ["d4_4"],
        "dim14": ["d4_5", "d4_6", "d4_7"],
        "dim15": ["d4_8", "d4_9", "d4_10"],
        "dim16": ["d5_1", "d5_2", "d5_3", "d5_4"],
        "dim17": ["d5_5", "d5_6"],
        "dim18": ["d5_7", "d5_8", "d5_9"],
        "dim19": ["d5_10"],
        "dim20": ["d6_1", "d6_2", "d6_3", "d6_4", "d6_5"],
        "dim21": ["d6_6"],
        "dim22": ["d6_7", "d6_8"],
        "dim23": ["d6_9", "d6_10"]
    }
    SOCIAL_VARIABLES = ['age', 'language', 'duree_emploi', 'type_emploi', 'organisation', 'pratique', 'genre']
    ANALYTICS_RESUME = []
    main()
    resume = "\n".join([element['description'] for element in ANALYTICS_RESUME])
    with open("plan.txt", "w", encoding="utf-8") as f:
        f.write(resume)
