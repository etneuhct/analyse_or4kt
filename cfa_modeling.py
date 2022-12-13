import itertools

import numpy
import pandas
from numpy.linalg import eigvals
from semopy import Model, calc_stats, report
from semopy.inspector import inspect
from statsmodels.stats.moment_helpers import cov2corr

from utils import add_result_for_export, export_to_excel


class CfaModel:

    def __init__(self, initial_dataset: pandas.DataFrame, structure, estimator, replacement_method, threshold=None):
        self.initial_dataset = initial_dataset
        self.structure = structure
        self.estimator = estimator
        self.replacement_method = replacement_method

        self.dataframe = initial_dataset.copy()
        self.str_structure = self._get_str_model()

        self.statistics = {}
        self.factor_loadings = None
        self.optimizations = []
        self.covariances = None
        self.threshold = threshold
        self.variables = self._get_used_variables()
        self.model = None
        self.eigen_value = None

    def run_analysis(self):
        self._reshape_data()
        dataframe = self.dataframe
        str_model = self.str_structure
        model = Model(str_model)
        model.fit(dataframe, obj=self.estimator)
        self.model = model

        self._set_statistics(model)
        self._set_factor_loadings(model)
        self._set_covariances(model)
        self._set_eigenvalue()
        self._run_optimization()
        self._set_summary()

    def _run_optimization(self):
        for threshold in range(1, 5):
            new_structure = self._optimized_factor_loadings(threshold * 0.1)
            if new_structure:
                self.optimizations.append({
                    "initial_dataset": self.initial_dataset,
                    "structure": new_structure,
                    "estimator": self.estimator,
                    "replacement_method": self.replacement_method,
                    "threshold": threshold
                })

    def _set_summary(self):
        self.summary = {
            "structure": self.str_structure,
            "estimator": self.estimator,
            "replacement_method": self.replacement_method,
            "variables": self.variables,
            "factor_loadings": self.factor_loadings,
            "statistics": self.statistics,
            "threshold": self.threshold,
            "model": self.model,
            "s_samples": self.model.n_samples,
            "fit": len(set([i > 0 for i in self.eigen_value])) == 1
        }

    def _get_used_variables(self):
        variables = []
        for element in self.structure.values():
            variables += element
        return variables

    def _get_str_model(self):
        str_model = "\n".join([f"{key} =~ {' + '.join(self.structure[key])}" for key in self.structure])
        return str_model

    def _reshape_data(self):
        self._set_data_columns()
        self._set_data_without_missing_values()

    def _set_data_without_missing_values(self):
        if self.replacement_method == 'remove':
            self.dataframe.dropna(how="any", inplace=True)
        elif self.replacement_method in ['mean', 'median']:

            for variable in self.dataframe.columns:
                if self.replacement_method == 'mean':
                    new_value = self.dataframe[variable].mean()
                else:
                    new_value = self.dataframe[variable].median()
                self.dataframe[variable].replace({numpy.nan, new_value}, inplace=True)

    def _set_data_columns(self):
        self.dataframe = self.dataframe[self.variables]

    def _set_statistics(self, cfa_model):
        stats = calc_stats(cfa_model)
        stats = stats.to_dict(orient='list')
        self.statistics = {key: stats[key][0] for key in stats}

    def _set_factor_loadings(self, cfa_model):
        inspection = inspect(cfa_model, std_est=True)
        factor_loadings = inspection[inspection['op'] == '~']
        self.factor_loadings = factor_loadings

    def _set_covariances(self, cfa_model):
        inspection = inspect(cfa_model, std_est=True, se_robust=True)
        dims = self.structure.keys()
        cov = inspection[inspection['op'] == '~~']
        cov = cov[cov['lval'].isin(dims)]
        self.covariances = cov

    def _set_eigenvalue(self):
        dims = self.structure.keys()
        cov = self.covariances
        df = pandas.DataFrame([[j for j in range(len(dims))] for _ in range(len(dims))], index=dims, columns=dims)
        for i in dims:
            for col in dims:
                value = None
                if not value:
                    s = cov[cov['lval'] == col]
                    s = s[s['rval'] == i]['Est. Std'].values
                    value = s[0] if len(s) > 0 else None
                if not value:
                    s = cov[cov['rval'] == col]
                    s = s[s['lval'] == i]['Est. Std'].values
                    value = s[0] if len(s) > 0 else None
                df.loc[i, col] = value
        corr = cov2corr(df)
        self.eigen_value = eigvals(corr)

    def _optimized_factor_loadings(self, threshold):
        remove_variables = []
        for i, row in self.factor_loadings.iterrows():
            est = row['Est. Std']
            variable = row['lval']
            if est < threshold:
                remove_variables.append(variable)
        if len(remove_variables) > 0:
            new_structure = {}
            for key in self.structure:
                new_variables = [i for i in self.structure[key] if i not in remove_variables]
                if len(new_variables) < 3:
                    pass
                else:
                    new_structure[key] = new_variables
            return new_structure


def export_cfa_results(cfa_results):
    factor_loadings_results = []
    summaries = []
    count = 1
    fls = []
    for element in cfa_results:
        model_name = f"Model #{count}"
        data = {
            "Model": model_name,
            "Nombres d'observation": element['s_samples'],
            "Nombre de colonnes": len(element['variables']),
            **element['statistics'], "Variables utilisées": ",".join(element['variables']),
            "Méthode de remplacement": element['replacement_method'],
            "Estimateur": element["estimator"],
            "Valeur minimum des facteurs": element['threshold'],
            "Structure du modèle": element['structure'],
            "Eigenvalues positives": element["fit"]}
        summaries.append(data)
        report(element['model'], model_name, 'results/cfa_models')
        add_result_for_export(fls,
                              [{'data': element['factor_loadings'], "name": model_name, "note": element['structure']}])
        count += 1
    summaries_df = pandas.DataFrame.from_records(summaries)
    add_result_for_export(factor_loadings_results, [{'data': summaries_df, "name": 'summary'}])
    add_result_for_export(factor_loadings_results, fls)
    export_to_excel(factor_loadings_results, file_name="results/cfa_results.xlsx")


def compute_cfa_models(dataframe):
    structure_1 = {
        "d1": ["d1_1", "d1_10", "d1_2", "d1_3", "d1_4", "d1_5", "d1_6", "d1_7", "d1_8", "d1_9"],
        "d2": ["d2_1", "d2_10", "d2_2", "d2_3", "d2_4", "d2_5", "d2_6", "d2_7", "d2_8", "d2_9"],
        "d3": ["d3_1", "d3_2", "d3_3", "d3_4", "d3_5", "d3_6", "d3_7", "d3_8", "d3_9"],
        "d4": ["d4_1", "d4_10", "d4_2", "d4_3", "d4_4", "d4_5", "d4_6", "d4_7", "d4_8", "d4_9"],
        "d5": ["d5_1", "d5_10", "d5_2", "d5_3", "d5_4", "d5_5", "d5_6", "d5_7", "d5_8", "d5_9"],
        "d6": ["d6_1", "d6_10", "d6_2", "d6_3", "d6_4", "d6_5", "d6_6", "d6_7", "d6_8", "d6_9"],
    }
    structure_2 = {
        "d1": ["d1_1", "d1_10", "d1_2", "d1_3", "d1_4", "d1_5", "d1_6", "d1_7", "d1_8", "d1_9"],
        "d2": ["d2_1", "d2_10", "d2_2", "d2_3", "d2_4", "d2_5", "d2_6", "d2_7", "d2_8", "d2_9"],
        "d3": ["d3_1", "d3_2", "d3_3", "d3_4", "d3_5", "d3_6", "d3_7", "d3_8", "d3_9"],
        "d4": ["d4_1", "d4_10", "d4_2", "d4_3", "d4_4", "d4_5", "d4_6", "d4_7", "d4_8", "d4_9"],
        "d5": ["d5_1", "d5_10", "d5_2", "d5_3", "d5_4", "d5_5", "d5_6", "d5_7", "d5_8", "d5_9"],
        "d6": ['dim20_prime', 'dim21_prime', 'dim22_prime', 'dim23_prime'],
    }
    structure_3 = {
        "d1": ["dim1_prime", "dim2_prime", "dim3_prime", "dim4_prime"],
        "d2": ["dim5_prime", "dim6_prime", "dim7_prime"],
        "d3": ["dim8_prime", "dim9_prime", "dim10_prime", "dim11_prime"],
        "d4": ["dim12_prime", "dim13_prime", "dim14_prime", "dim15_prime"],
        "d5": ["dim16_prime", "dim17_prime", "dim18_prime", "dim19_prime"],
        "d6": ['dim20_prime', 'dim21_prime', 'dim22_prime', 'dim23_prime'],
    }
    structure_4 = {
        "d1": ["dim1_prime", "dim2_prime", "dim3_prime", "dim4_prime"],
        "d2": ["dim5_prime", "dim6_prime", "dim7_prime"],
        "d3": ["dim8_prime", "dim9_prime", "dim10_prime", "dim11_prime"],
        "d4": ["dim12_prime", "dim13_prime", "dim14_prime", "dim15_prime"],
        "d5": ["dim16_prime", "dim17_prime", "dim18_prime", "dim19_prime"]
    }
    structure_5 = {
        "d1": ["d1_1", "d1_10", "d1_2", "d1_3", "d1_4", "d1_5", "d1_6", "d1_7", "d1_8", "d1_9"],
        "d2": ["d2_1", "d2_10", "d2_2", "d2_3", "d2_4", "d2_5", "d2_6", "d2_7", "d2_8", "d2_9"],
        "d3": ["d3_1", "d3_2", "d3_3", "d3_4", "d3_5", "d3_6", "d3_7", "d3_8", "d3_9"],
        "d4": ["d4_1", "d4_10", "d4_2", "d4_3", "d4_4", "d4_5", "d4_6", "d4_7", "d4_8", "d4_9"],
        "d5": ["d5_1", "d5_10", "d5_2", "d5_3", "d5_4", "d5_5", "d5_6", "d5_7", "d5_8", "d5_9"]
    }
    # structure_6 = {
    #     "dim1": ["d1_1", "d1_2", "d1_3"],
    #     "dim2": ["d1_4", "d1_5"],
    #     "dim3": ["d1_6", "d1_7", "d1_8"],
    #     "dim4": ["d1_9", "d1_10"],
    #     "dim5": ["d2_1"],
    #     "dim6": ["d2_2", "d2_3", "d2_4", "d2_5"],
    #     "dim7": ["d2_6", "d2_7", "d2_8", "d2_9", "d2_10"],
    #     "dim8": ["d3_1", "d3_2"],
    #     "dim9": ["d3_3", "d3_4", "d3_5"],
    #     "dim10": ["d3_6", "d3_7", "d3_8"],
    #     "dim11": ["d3_9"],
    #     "dim12": ["d4_1", "d4_2", "d4_3"],
    #     "dim13": ["d4_4"],
    #     "dim14": ["d4_5", "d4_6", "d4_7"],
    #     "dim15": ["d4_8", "d4_9", "d4_10"],
    #     "dim16": ["d5_1", "d5_2", "d5_3", "d5_4"],
    #     "dim17": ["d5_5", "d5_6"],
    #     "dim18": ["d5_7", "d5_8", "d5_9"],
    #     "dim19": ["d5_10"],
    #     "dim20": ["d6_1", "d6_2", "d6_3", "d6_4", "d6_5"],
    #     "dim21": ["d6_6"],
    #     "dim22": ["d6_7", "d6_8"],
    #     "dim23": ["d6_9", "d6_10"]
    # }

    data = {
        "initial_dataset": [dataframe],
        "structure": [structure_1, structure_2, structure_3, structure_4, structure_5],
        "estimator": ['MLW'],  # 'ULS', 'GLS', 'WLS', 'DWLS'],
        "replacement_method": ['remove', 'mean'],
    }

    combinations = itertools.product(*data.values())
    all_options = [{list(data.keys())[i]: combination[i] for i in range(len(data.keys()))} for combination in
                   combinations]
    count = 0
    cfa_results = []
    for option in all_options:
        cfa_model = CfaModel(**option)
        cfa_model.run_analysis()
        all_options += cfa_model.optimizations
        cfa_results.append(cfa_model.summary)
        count += 1
    export_cfa_results(cfa_results)
