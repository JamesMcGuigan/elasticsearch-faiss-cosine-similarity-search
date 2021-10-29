import pycountry

# BUGFIX: manual lookup from experiments/udhr/reports/fasttext_predict_udhr.log
language_code_overrides = {
    "alpha_2": {
        "greek":     "gr",
        "slovene":   "sl",
        "hungarian": "hu",
    }
}
language_name_overrides = {
    "en": "English",
    "hu": "Hungarian",
}

def get_language_code(name, type='alpha_2') -> str:
    """
    This is a wrapper function around pycountry.languages.lookup().alpha_2, 
    with manual overrides
    Question: should we be using alpha_2 or alpha_3?
    Examples:
        Language(
            alpha_2='cs', 
            alpha_3='ces', 
            bibliographic='cze', 
            name='Czech', 
            scope='I', 
            type='L'
        )
        Language(
            alpha_2='da', alpha_3='dan', name='Danish', scope='I', type='L'
        )
    """
    if not name: 
        return ''

    code1 = (  language_code_overrides.get(type, {}).get(name.lower(), '')
            or language_code_overrides.get(type, {}).get(get_language_name(name).lower(), '') )
    code2 = ''
    code3 = name.lower() if len(name) == 2 else ''
    try:
        lang  = pycountry.languages.lookup(name)
        code2 = getattr(lang, type, '')
    except (LookupError, KeyError): 
        pass

    if code1 and code2 and code1 != code2: 
        print(f'WARNING: languages_lookup({name}, {type}): '
              f'pycountry.languages.lookup() -> {code1} != '
              f'language_code_overrides[][] -> {code2}')
    if not code1 and not code2:
        print(
            f'ERROR:   languages_lookup({name}, {type}): unknown name: {name}'
        )
    # prefer override, then pycountry, then name.lower()
    return code1 or code2 or code3

def get_language_name(code) -> str:
    assert isinstance(code, str)
    name = code
    try:
        if code.lower() in language_name_overrides:
            name = language_name_overrides[code.lower()]
        else:
            name = pycountry.languages.lookup(code)
            name = getattr(name, 'name', '')
    except LookupError: pass
    return name
