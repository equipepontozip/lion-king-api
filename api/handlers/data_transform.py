import pandas as pd
import sys


def array_for_predict(req_json):
    print('ayn')
    #array precisa ser de 31, e preencher com 0 o que não tiver.
    return [0,1,0,1]


def transform_keystroke(data):
    df = pd.DataFrame.from_dict([data])
    df.fillna(0, inplace=True)
    print(df.columns)

    df['DD.5.shift.r'] = df.get('DD.5.shift', 0) + df.get('DD.shift.r', 0)
    df['UD.5.shift.r'] = df.get('UD.5.r', 0) + df.get('UD.5.shift', 0)

    df['H.shift.r'] = df.get('H.r', 0)
    df['DD.shift.r.o'] = df.get('DD.r.o', 0)
    df['UD.shift.r.o'] = df.get('UD.r.o', 0) + df.get('UD.shift.o', 0)

    df.drop(columns=['UD.r.o', 'UD.shift.o', 'H.r', 'UD.5.r', 'UD.5.shift', 'DD.shift.r', 'DD.5.shift'], inplace=True, errors='ignore')
    df['H.return'] = df['H.l']

    colsnew = ['H.period', 'DD.period.t', 'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 'UD.i.e',
               'H.e', 'DD.e.5', 'UD.e.5', 'H.5', 'DD.5.shift.r', 'UD.5.shift.r', 'H.shift.r', 'DD.shift.r.o',
               'UD.shift.r.o', 'H.o', 'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n',
               'DD.n.l', 'UD.n.l', 'H.l', 'DD.l.return', 'UD.l.return', 'H.return'
    ]

    try:
	    df = df[colsnew]
    except KeyError:
	    return False


    return df
