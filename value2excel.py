import pandas as pd


def getExcel(features, excel_path, header=True):
    """保存values到excel文件中"""
    data = pd.DataFrame(features)
    writer = pd.ExcelWriter(excel_path)
    if header:
        data.to_excel(writer, 'page_1', float_format='%.5f')
    else:
        data.to_excel(writer, 'page_1', float_format='%.5f', header=False, index=False)
    writer.save()
    writer.close()
