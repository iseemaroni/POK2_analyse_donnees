import pandas

pandas.__version__

prénom = pandas.read_csv("nat2022_csv.zip",
                         compression='zip',
                         sep=";")