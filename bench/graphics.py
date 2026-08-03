import fpy2 as fp


@fp.fpy(
    meta={
        "name": "An eigenvalue calculation from TNG",
        "fpbench_domain": "graphics",
    }
)
def eigenvalue_calculation_TNG(
    a: fp.Real,
    b: fp.Real,
    c: fp.Real,
    d: fp.Real,
) -> fp.Real:
    return fp.sqrt(
        fp.pow(a + d, fp.round(2)) + fp.pow(b - c, fp.round(2))
    ) - fp.sqrt(fp.pow(a - d, fp.round(2)) + fp.pow(b + c, fp.round(2)))


@fp.fpy(
    meta={
        "name": "An eigenvalue calculation from TNG",
        "description": "This version is much more numerically stable.",
        "fpbench_domain": "graphics",
    }
)
def eigenvalue_calculation_TNG_stable(
    a: fp.Real,
    b: fp.Real,
    c: fp.Real,
    d: fp.Real,
) -> fp.Real:
    sos = ((a * a + b * b) + c * c) + d * d
    det = a * d - b * c
    return fp.sqrt(sos + fp.round(2) * det) - fp.sqrt(
        sos - fp.round(2) * det
    )


@fp.fpy(
    meta={
        "name": "An eigenvalue calculation from TNG",
        "description": "This version is best if sos >> det.",
        "fpbench_domain": "graphics",
    }
)
def eigenvalue_calculation_TNG_sos_dominates_det(
    a: fp.Real,
    b: fp.Real,
    c: fp.Real,
    d: fp.Real,
) -> fp.Real:
    sos = ((a * a + b * b) + c * c) + d * d
    det = a * d - b * c
    return fp.round(2) * det / fp.sqrt(sos)
