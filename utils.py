import io
import itertools

import pandas


def export_to_excel(results, file_name='results/global_result.xlsx'):
    file = io.BytesIO()
    writer = pandas.ExcelWriter(file, engine='xlsxwriter')

    for result in results:
        data = result['data']
        name = result['name']
        data.to_excel(writer, sheet_name=name, startrow=3)
        note = result['note'] if 'note' in result else None
        if note:
            worksheet = writer.sheets[name]
            worksheet.write(0, 0, note)
    writer.save()
    with open(file_name, 'wb') as f:
        f.write(file.getbuffer())


def add_result_for_export(result_list, new_results):
    for new_result in new_results:
        result_list.append(new_result)


def all_subsets(initial_list, minimum=0, maximum=None):
    result = []
    maximum = len(initial_list) + 1 if maximum is None else maximum
    for L in range(minimum, maximum):
        for subset in itertools.combinations(initial_list, L):
            result.append(subset)
    return result