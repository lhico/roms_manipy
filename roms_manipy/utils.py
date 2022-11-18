import ast

def get_dict_paths(fpath: str):
    """Read a text file with a dictionary and returns
    the dictionary"""
    file = open(fpath, 'r')
    contents = file.read()
    dictionary = ast.literal_eval(contents)
    return dictionary

def crop_wesn(w: float, e: float, s: float, n: float,
              dtw: float=0.,dte: float=0.,
              dts: float=0.,dtn: float=0.):
    """creates a list of [w,e,s,n] and a second list with
       a cropped/extended version [w+dtw, e+dte, s+dts, n+dtn]

    Args:
        w (float): western value
        e (float): eastern value
        s (float): southern value
        n (float): northern value
        dtw (int, optional): western value offset. Defaults to 0.
        dte (int, optional): eastern value offset. Defaults to 0.
        dts (int, optional): southern value offset. Defaults to 0.
        dtn (int, optional): northern value offset. Defaults to 0.

    Returns:
        _type_: _description_
    """

    wesn = [w, e, s ,n]
    wesnc = [w+dtw, e+dte, s+dts ,n+dtn]
    return wesn, wesnc
    