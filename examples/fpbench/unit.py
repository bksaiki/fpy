from fpy2 import *

@fpy(
    meta={
        'name': 'Test < (1/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60___u40_1_u47_9_u41_():
    if round(1.0) < round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (2/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60___u40_2_u47_9_u41_():
    if round(0.0) < round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (3/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60___u40_3_u47_9_u41_():
    if round(-1.0) < round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (4/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60___u40_4_u47_9_u41_():
    if round(0.0) < round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (5/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60___u40_5_u47_9_u41_():
    if round(1.0) < round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (6/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60___u40_6_u47_9_u41_():
    if round(-1.0) < round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (7/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60___u40_7_u47_9_u41_():
    if round(0.0) < round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (8/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60___u40_8_u47_9_u41_():
    if nan() < nan():
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test < (9/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60___u40_9_u47_9_u41_():
    if round(0.0) < round(0.0) < round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (1/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62___u40_1_u47_9_u41_():
    if round(1.0) > round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (2/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62___u40_2_u47_9_u41_():
    if round(0.0) > round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (3/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62___u40_3_u47_9_u41_():
    if round(-1.0) > round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (4/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62___u40_4_u47_9_u41_():
    if round(0.0) > round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (5/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62___u40_5_u47_9_u41_():
    if round(1.0) > round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (6/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62___u40_6_u47_9_u41_():
    if round(-1.0) > round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (7/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62___u40_7_u47_9_u41_():
    if round(0.0) > round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (8/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62___u40_8_u47_9_u41_():
    if nan() > nan():
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test > (9/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62___u40_9_u47_9_u41_():
    if round(0.0) > round(0.0) > round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (1/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60__u61___u40_1_u47_9_u41_():
    if round(1.0) <= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (2/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60__u61___u40_2_u47_9_u41_():
    if round(0.0) <= round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (3/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60__u61___u40_3_u47_9_u41_():
    if round(-1.0) <= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (4/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60__u61___u40_4_u47_9_u41_():
    if round(0.0) <= round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (5/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60__u61___u40_5_u47_9_u41_():
    if round(1.0) <= round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (6/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60__u61___u40_6_u47_9_u41_():
    if round(-1.0) <= round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (7/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60__u61___u40_7_u47_9_u41_():
    if round(0.0) <= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (8/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u60__u61___u40_8_u47_9_u41_():
    if nan() <= nan():
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test <= (9/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u60__u61___u40_9_u47_9_u41_():
    if round(0.0) <= round(0.0) <= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (1/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62__u61___u40_1_u47_9_u41_():
    if round(1.0) >= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (2/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62__u61___u40_2_u47_9_u41_():
    if round(0.0) >= round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (3/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62__u61___u40_3_u47_9_u41_():
    if round(-1.0) >= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (4/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62__u61___u40_4_u47_9_u41_():
    if round(0.0) >= round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (5/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62__u61___u40_5_u47_9_u41_():
    if round(1.0) >= round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (6/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62__u61___u40_6_u47_9_u41_():
    if round(-1.0) >= round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (7/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62__u61___u40_7_u47_9_u41_():
    if round(0.0) >= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (8/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u62__u61___u40_8_u47_9_u41_():
    if nan() >= nan():
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test >= (9/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u62__u61___u40_9_u47_9_u41_():
    if round(0.0) >= round(0.0) >= round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (1/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_1_u47_9_u41_():
    if round(1.0) == round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (2/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_2_u47_9_u41_():
    if round(0.0) == round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (3/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_3_u47_9_u41_():
    if round(-1.0) == round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (4/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_4_u47_9_u41_():
    if round(0.0) == round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (5/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_5_u47_9_u41_():
    if round(1.0) == round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (6/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_6_u47_9_u41_():
    if round(-1.0) == round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (7/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u61__u61___u40_7_u47_9_u41_():
    if round(0.0) == round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (8/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u61__u61___u40_8_u47_9_u41_():
    if nan() == nan():
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test == (9/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u61__u61___u40_9_u47_9_u41_():
    if round(0.0) == round(0.0) == round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (1/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_1_u47_9_u41_():
    if round(1.0) != round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (2/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_2_u47_9_u41_():
    if round(0.0) != round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (3/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_3_u47_9_u41_():
    if round(-1.0) != round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (4/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_4_u47_9_u41_():
    if round(0.0) != round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (5/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_5_u47_9_u41_():
    if round(1.0) != round(-1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (6/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_6_u47_9_u41_():
    if round(-1.0) != round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (7/9)',
        'spec': lambda : round(0.0),
    }
)
def Test__u33__u61___u40_7_u47_9_u41_():
    if round(0.0) != round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (8/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_8_u47_9_u41_():
    if nan() != nan():
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test != (9/9)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33__u61___u40_9_u47_9_u41_():
    if round(0.0) == round(0.0) == round(0.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test and (1/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_and__u40_1_u47_6_u41_():
    if True and True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test and (2/6)',
        'spec': lambda : round(0.0),
    }
)
def Test_and__u40_2_u47_6_u41_():
    if True and False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test and (3/6)',
        'spec': lambda : round(0.0),
    }
)
def Test_and__u40_3_u47_6_u41_():
    if False and True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test and (4/6)',
        'spec': lambda : round(0.0),
    }
)
def Test_and__u40_4_u47_6_u41_():
    if False and False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test and (5/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_and__u40_5_u47_6_u41_():
    if True and True and True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test and (6/6)',
        'spec': lambda : round(0.0),
    }
)
def Test_and__u40_6_u47_6_u41_():
    if True and True and False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test or (1/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_or__u40_1_u47_4_u41_():
    if True or True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test or (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_or__u40_2_u47_4_u41_():
    if True or False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test or (3/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_or__u40_3_u47_4_u41_():
    if False or True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test or (4/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_or__u40_4_u47_4_u41_():
    if False or False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test or (5/6)',
        'spec': lambda : round(0.0),
    }
)
def Test_or__u40_5_u47_6_u41_():
    if False or False or False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test or (6/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_or__u40_6_u47_6_u41_():
    if False or False or True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test not (1/2)',
        'spec': lambda : round(0.0),
    }
)
def Test_not__u40_1_u47_2_u41_():
    if not True:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test not (2/2)',
        'spec': lambda : round(1.0),
    }
)
def Test_not__u40_2_u47_2_u41_():
    if not False:
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isinf (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isinf__u40_1_u47_4_u41_():
    if isinf(round(0.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isinf (2/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isinf__u40_2_u47_4_u41_():
    if isinf(round(1.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isinf (3/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_isinf__u40_3_u47_4_u41_():
    if isinf(inf()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isinf (4/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isinf__u40_4_u47_4_u41_():
    if isinf(nan()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnan (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isnan__u40_1_u47_4_u41_():
    if isnan(round(0.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnan (2/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isnan__u40_2_u47_4_u41_():
    if isnan(round(1.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnan (3/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isnan__u40_3_u47_4_u41_():
    if isnan(inf()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnan (4/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_isnan__u40_4_u47_4_u41_():
    if isnan(nan()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isfinite (1/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_isfinite__u40_1_u47_4_u41_():
    if isfinite(round(0.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isfinite (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_isfinite__u40_2_u47_4_u41_():
    if isfinite(round(1.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isfinite (3/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isfinite__u40_3_u47_4_u41_():
    if isfinite(inf()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isfinite (4/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isfinite__u40_4_u47_4_u41_():
    if isfinite(nan()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnormal (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isnormal__u40_1_u47_4_u41_():
    if isnormal(round(0.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnormal (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_isnormal__u40_2_u47_4_u41_():
    if isnormal(round(1.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnormal (3/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isnormal__u40_3_u47_4_u41_():
    if isnormal(inf()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test isnormal (4/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_isnormal__u40_4_u47_4_u41_():
    if isnormal(nan()):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test signbit (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_signbit__u40_1_u47_3_u41_():
    if signbit(round(0.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test signbit (2/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_signbit__u40_2_u47_3_u41_():
    if signbit(round(1.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test signbit (3/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_signbit__u40_3_u47_3_u41_():
    if signbit(round(-1.0)):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test E (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_E__u40_1_u47_1_u41_():
    if round(2.0) < const_e() and const_e() < round(3.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test LOG2E (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_LOG2E__u40_1_u47_1_u41_():
    if round(1.0) < const_log2e() and const_log2e() < round(2.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test LOG10E (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_LOG10E__u40_1_u47_1_u41_():
    if round(0.25) < const_log10e() and const_log10e() < round(0.5):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test LN2 (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_LN2__u40_1_u47_1_u41_():
    if round(0.5) < const_ln2() and const_ln2() < round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test PI (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_PI__u40_1_u47_1_u41_():
    if round(3.0) < const_pi() and const_pi() < round(4.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test PI_2 (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_PI_2__u40_1_u47_1_u41_():
    if round(1.0) < const_pi_2() and const_pi_2() < round(2.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test PI_4 (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_PI_4__u40_1_u47_1_u41_():
    if round(0.5) < const_pi_4() and const_pi_4() < round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test M_1_PI (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_M_1_PI__u40_1_u47_1_u41_():
    if round(0.25) < const_1_pi() and const_1_pi() < round(0.5):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test M_2_PI (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_M_2_PI__u40_1_u47_1_u41_():
    if round(0.5) < const_2_pi() and const_2_pi() < round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test M_2_SQRTPI (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_M_2_SQRTPI__u40_1_u47_1_u41_():
    if round(1.0) < const_2_sqrtpi() and const_2_sqrtpi() < round(2.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test SQRT2 (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_SQRT2__u40_1_u47_1_u41_():
    if round(1.0) < const_sqrt2() and const_sqrt2() < round(2.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test SQRT1_2 (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_SQRT1_2__u40_1_u47_1_u41_():
    if round(0.5) < const_sqrt1_2() and const_sqrt1_2() < round(1.0):
        t = round(1)
    else:
        t = round(0)
    return t

@fpy(
    meta={
        'name': 'Test let (1/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_let__u40_1_u47_5_u41_():
    a = round(1.0)
    return a

@fpy(
    meta={
        'name': 'Test let (2/5)',
        'spec': lambda : round(2.0),
    }
)
def Test_let__u40_2_u47_5_u41_():
    a = round(1.0)
    b = round(1.0)
    return (a + b)

@fpy(
    meta={
        'name': 'Test let (3/5)',
        'spec': lambda : round(-1.0),
    }
)
def Test_let__u40_3_u47_5_u41_():
    a = round(1.0)
    b = round(0.0)
    c = round(-1.0)
    d = round(0)
    if a > b:
        t = c
    else:
        t = d
    return t

@fpy(
    meta={
        'name': 'Test let (4/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_let__u40_4_u47_5_u41_():
    a = round(1.0)
    b = round(0.0)
    a0 = b
    b1 = a
    return b1

@fpy(
    meta={
        'name': 'Test let (5/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_let__u40_5_u47_5_u41_():
    a = round(1.0)
    b = round(0.0)
    a0 = b
    b1 = a
    a2 = b1
    b3 = a0
    return b3

@fpy(
    meta={
        'name': 'Test let* (1/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_let_u42___u40_1_u47_5_u41_():
    a = round(1.0)
    return a

@fpy(
    meta={
        'name': 'Test let* (2/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_let_u42___u40_2_u47_5_u41_():
    a = round(1.0)
    b = round(-1.0)
    return (a + b)

@fpy(
    meta={
        'name': 'Test let* (3/5)',
        'spec': lambda : round(-1.0),
    }
)
def Test_let_u42___u40_3_u47_5_u41_():
    a = round(1.0)
    b = round(0.0)
    c = round(-1.0)
    d = round(0)
    if a > b:
        t = c
    else:
        t = d
    return t

@fpy(
    meta={
        'name': 'Test let* (4/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_let_u42___u40_4_u47_5_u41_():
    a = round(1.0)
    b = a
    return b

@fpy(
    meta={
        'name': 'Test let* (5/5)',
        'spec': lambda : round(5.0),
    }
)
def Test_let_u42___u40_5_u47_5_u41_():
    a = round(3.0)
    b = round(1.0)
    c = round(-1.0)
    b0 = a
    c1 = (c + (b0 + a))
    a2 = c1
    b3 = a2
    return b3

@fpy(
    meta={
        'name': 'Test while (1/7)',
        'spec': lambda : round(0.0),
    }
)
def Test_while__u40_1_u47_7_u41_():
    a = round(0.0)
    while False:
        t = (a + round(1.0))
        a = t
    return a

@fpy(
    meta={
        'name': 'Test while (2/7)',
        'spec': lambda : round(4.0),
    }
)
def Test_while__u40_2_u47_7_u41_():
    a = round(0.0)
    while a < round(4):
        t = (a + round(1.0))
        a = t
    return a

@fpy(
    meta={
        'name': 'Test while (3/7)',
        'spec': lambda : round(8.0),
    }
)
def Test_while__u40_3_u47_7_u41_():
    a = round(0.0)
    b = round(1.0)
    while a < round(3):
        t = (a + round(1.0))
        t0 = (b * round(2.0))
        a = t
        b = t0
    return b

@fpy(
    meta={
        'name': 'Test while (4/7)',
        'spec': lambda : round(-6.0),
    }
)
def Test_while__u40_4_u47_7_u41_():
    a = round(0.0)
    b = round(0.0)
    while a <= round(3):
        t = (a + round(1.0))
        t0 = (b - a)
        a = t
        b = t0
    return b

@fpy(
    meta={
        'name': 'Test while (5/7)',
        'spec': lambda : round(1.0),
    }
)
def Test_while__u40_5_u47_7_u41_():
    i = round(0)
    a = round(0.0)
    b = round(1.0)
    while i < round(3):
        t = (i + round(1))
        t0 = b
        t1 = a
        i = t
        a = t0
        b = t1
    return a

@fpy(
    meta={
        'name': 'Test while (6/7)',
        'spec': lambda : round(0.0),
    }
)
def Test_while__u40_6_u47_7_u41_():
    i = round(0)
    a = round(0.0)
    b = round(1.0)
    while i < round(4):
        t = (i + round(1))
        t0 = b
        t1 = a
        i = t
        a = t0
        b = t1
    return a

@fpy(
    meta={
        'name': 'Test while (7/7)',
        'spec': lambda : round(6.0),
    }
)
def Test_while__u40_7_u47_7_u41_():
    a = round(0.0)
    b = round(0.0)
    while a <= round(3):
        t = (a + round(1.0))
        i = round(0)
        x = round(0.0)
        while i <= a:
            t0 = (i + round(1))
            t1 = (x + i)
            i = t0
            x = t1
        t2 = x
        a = t
        b = t2
    return b

@fpy(
    meta={
        'name': 'Test while* (1/7)',
        'spec': lambda : round(0.0),
    }
)
def Test_while_u42___u40_1_u47_7_u41_():
    a = round(0.0)
    while False:
        a = (a + round(1.0))
    return a

@fpy(
    meta={
        'name': 'Test while* (2/7)',
        'spec': lambda : round(4.0),
    }
)
def Test_while_u42___u40_2_u47_7_u41_():
    a = round(0.0)
    while a < round(4):
        a = (a + round(1.0))
    return a

@fpy(
    meta={
        'name': 'Test while* (3/7)',
        'spec': lambda : round(8.0),
    }
)
def Test_while_u42___u40_3_u47_7_u41_():
    a = round(0.0)
    b = round(1.0)
    while a < round(3):
        a = (a + round(1.0))
        b = (b * round(2.0))
    return b

@fpy(
    meta={
        'name': 'Test while* (4/7)',
        'spec': lambda : round(-10.0),
    }
)
def Test_while_u42___u40_4_u47_7_u41_():
    a = round(0.0)
    b = round(0.0)
    while a <= round(3):
        a = (a + round(1.0))
        b = (b - a)
    return b

@fpy(
    meta={
        'name': 'Test while* (5/7)',
        'spec': lambda : round(1.0),
    }
)
def Test_while_u42___u40_5_u47_7_u41_():
    i = round(0.0)
    a = round(0.0)
    b = round(1.0)
    while i < round(3):
        i = (i + round(1))
        a = b
        b = a
    return a

@fpy(
    meta={
        'name': 'Test while* (6/7)',
        'spec': lambda : round(1.0),
    }
)
def Test_while_u42___u40_6_u47_7_u41_():
    i = round(0.0)
    a = round(0.0)
    b = round(1.0)
    while i < round(4):
        i = (i + round(1))
        a = b
        b = a
    return a

@fpy(
    meta={
        'name': 'Test while* (7/7)',
        'spec': lambda : round(15.0),
    }
)
def Test_while_u42___u40_7_u47_7_u41_():
    a = round(0.0)
    b = round(0.0)
    while a <= round(3):
        a = (a + round(1.0))
        i = round(0)
        x = round(0.0)
        while i <= a:
            i = (i + round(1))
            x = (x + i)
        b = x
    return b

@fpy(
    meta={
        'name': 'Test if (1/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_if__u40_1_u47_6_u41_():
    if round(1.0) > round(0.0):
        t = round(1.0)
    else:
        t = round(0.0)
    return t

@fpy(
    meta={
        'name': 'Test if (2/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_if__u40_2_u47_6_u41_():
    if round(1.0) < round(0.0):
        t = round(0.0)
    else:
        t = round(1.0)
    return t

@fpy(
    meta={
        'name': 'Test if (3/6)',
    }
)
def Test_if__u40_3_u47_6_u41_():
    if True:
        y = round(1)
        t2 = y
    else:
        if False:
            y0 = round(2)
            t = y0
        else:
            y1 = round(3)
            t = y1
        t2 = t
    return t2

@fpy(
    meta={
        'name': 'Test if (4/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_if__u40_4_u47_6_u41_():
    if True:
        t1 = round(1)
    else:
        t = False
        if t:
            t0 = round(0)
        else:
            t0 = round(0)
        t1 = t0
    return t1

@fpy(
    meta={
        'name': 'Test if (5/6)',
    }
)
def Test_if__u40_5_u47_6_u41_():
    t = False
    if t:
        t4 = round(0)
    else:
        t0 = True
        if t0:
            t3 = round(1)
        else:
            t1 = True
            if t1:
                t2 = round(0)
            else:
                t2 = round(0)
            t3 = t2
        t4 = t3
    return t4

@fpy(
    meta={
        'name': 'Test if (6/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_if__u40_6_u47_6_u41_():
    if False:
        t1 = round(1.0)
    else:
        y = round(1)
        t = round(2) > y
        if t:
            t0 = round(1)
        else:
            t0 = round(0)
        t1 = t0
    return t1

@fpy(
    meta={
        'name': 'Test cast (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_cast__u40_1_u47_1_u41_():
    return round(round(1.0))

@fpy(
    meta={
        'name': 'Test ! (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test__u33___u40_1_u47_1_u41_():
    with IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE, overflow=OverflowMode.OVERFLOW, num_randbits=0):
        t = round(1.0)
    return t

@fpy(
    meta={
        'name': 'Test FPCore identifier (FPBench 2.0)',
        'spec': lambda : round(1.0),
    }
)
def ident():
    return round(1.0)

@fpy(
    meta={
        'name': 'Test arguments (1/2)',
    }
)
def Test_arguments__u40_1_u47_2_u41_(x):
    return x

@fpy(
    meta={
        'name': 'Test arguments (2/2)',
    }
)
def Test_arguments__u40_2_u47_2_u41_(x, y):
    return x

@fpy(
    meta={
        'name': 'Nested syntax (1/2)',
    }
)
def Nested_syntax__u40_1_u47_2_u41_(x):
    if x < round(0):
        t = round(-1)
    else:
        t = round(1)
    return (round(1) + t)

@fpy(
    meta={
        'name': 'Nested syntax (2/2)',
    }
)
def Nested_syntax__u40_2_u47_2_u41_(x):
    y = round(1)
    return (round(1) + (y + x))

@fpy(
    meta={
        'name': 'Test decnum (1/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_decnum__u40_1_u47_5_u41_():
    return round(0.0)

@fpy(
    meta={
        'name': 'Test decnum (2/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_decnum__u40_2_u47_5_u41_():
    return round(1.0)

@fpy(
    meta={
        'name': 'Test decnum (3/5)',
        'spec': lambda : round(-1.0),
    }
)
def Test_decnum__u40_3_u47_5_u41_():
    return round(-1.0)

@fpy(
    meta={
        'name': 'Test decnum (4/5)',
        'spec': lambda : round(1.5),
    }
)
def Test_decnum__u40_4_u47_5_u41_():
    return round(1.5)

@fpy(
    meta={
        'name': 'Test decnum (5/5)',
        'spec': lambda : round(0.75),
    }
)
def Test_decnum__u40_5_u47_5_u41_():
    return round(0.75)

@fpy(
    meta={
        'name': 'Test hexnum (1/6)',
        'spec': lambda : round(0.0),
    }
)
def Test_hexnum__u40_1_u47_6_u41_():
    return round(hexnum('0x0.0p+0'))

@fpy(
    meta={
        'name': 'Test hexnum (2/6)',
        'spec': lambda : round(1.0),
    }
)
def Test_hexnum__u40_2_u47_6_u41_():
    return round(hexnum('0x1.0p+0'))

@fpy(
    meta={
        'name': 'Test hexnum (3/6)',
        'spec': lambda : round(-1.0),
    }
)
def Test_hexnum__u40_3_u47_6_u41_():
    return round(hexnum('-0x1.0p+0'))

@fpy(
    meta={
        'name': 'Test hexnum (4/6)',
        'spec': lambda : round(0.5),
    }
)
def Test_hexnum__u40_4_u47_6_u41_():
    return round(hexnum('0x1.0p-1'))

@fpy(
    meta={
        'name': 'Test hexnum (5/6)',
        'spec': lambda : round(-2.0),
    }
)
def Test_hexnum__u40_5_u47_6_u41_():
    return round(hexnum('-0x1.0p+1'))

@fpy(
    meta={
        'name': 'Test hexnum (6/6)',
        'spec': lambda : round(0.99609375),
    }
)
def Test_hexnum__u40_6_u47_6_u41_():
    return round(hexnum('0xf.fp-4'))

@fpy(
    meta={
        'name': 'Test rational (1/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_rational__u40_1_u47_5_u41_():
    return round(0)

@fpy(
    meta={
        'name': 'Test rational (2/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_rational__u40_2_u47_5_u41_():
    return round(1)

@fpy(
    meta={
        'name': 'Test rational (3/5)',
        'spec': lambda : round(-1.0),
    }
)
def Test_rational__u40_3_u47_5_u41_():
    return round(-1)

@fpy(
    meta={
        'name': 'Test rational (4/5)',
        'spec': lambda : round(-0.5),
    }
)
def Test_rational__u40_4_u47_5_u41_():
    return round(rational(-1, 2))

@fpy(
    meta={
        'name': 'Test rational (5/5)',
        'spec': lambda : round(1.25),
    }
)
def Test_rational__u40_5_u47_5_u41_():
    return round(rational(5, 4))

@fpy(
    meta={
        'name': 'Test digits (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_digits__u40_1_u47_4_u41_():
    return round(digits(0, 0, 2))

@fpy(
    meta={
        'name': 'Test digits (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_digits__u40_2_u47_4_u41_():
    return round(digits(1, 0, 2))

@fpy(
    meta={
        'name': 'Test digits (3/4)',
        'spec': lambda : round(-2.0),
    }
)
def Test_digits__u40_3_u47_4_u41_():
    return round(digits(-1, 1, 2))

@fpy(
    meta={
        'name': 'Test digits (4/4)',
        'spec': lambda : round(1.5),
    }
)
def Test_digits__u40_4_u47_4_u41_():
    return round(digits(3, -1, 2))

@fpy(
    meta={
        'name': 'Test - (1/2)',
        'spec': lambda : round(-1.0),
    }
)
def Test____u40_1_u47_2_u41_():
    return -round(1.0)

@fpy(
    meta={
        'name': 'Test - (2/2)',
        'spec': lambda : round(1.0),
    }
)
def Test____u40_2_u47_2_u41_():
    return -round(-1.0)

@fpy(
    meta={
        'name': 'Test + (1/3)',
        'spec': lambda : round(1.0),
    }
)
def Test__u43___u40_1_u47_3_u41_():
    return (round(0.0) + round(1.0))

@fpy(
    meta={
        'name': 'Test + (2/3)',
        'spec': lambda : round(2.0),
    }
)
def Test__u43___u40_2_u47_3_u41_():
    return (round(1.0) + round(1.0))

@fpy(
    meta={
        'name': 'Test + (3/3)',
        'spec': lambda : round(0.0),
    }
)
def Test__u43___u40_3_u47_3_u41_():
    return (round(-1.0) + round(1.0))

@fpy(
    meta={
        'name': 'Test - (1/3)',
        'spec': lambda : round(-1.0),
    }
)
def Test____u40_1_u47_3_u41_():
    return (round(0.0) - round(1.0))

@fpy(
    meta={
        'name': 'Test - (2/3)',
        'spec': lambda : round(0.0),
    }
)
def Test____u40_2_u47_3_u41_():
    return (round(1.0) - round(1.0))

@fpy(
    meta={
        'name': 'Test - (3/3)',
        'spec': lambda : round(-2.0),
    }
)
def Test____u40_3_u47_3_u41_():
    return (round(-1.0) - round(1.0))

@fpy(
    meta={
        'name': 'Test * (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test__u42___u40_1_u47_3_u41_():
    return (round(0.0) * round(1.0))

@fpy(
    meta={
        'name': 'Test * (2/3)',
        'spec': lambda : round(-2.0),
    }
)
def Test__u42___u40_2_u47_3_u41_():
    return (round(2.0) * round(-1.0))

@fpy(
    meta={
        'name': 'Test * (3/3)',
        'spec': lambda : round(4.0),
    }
)
def Test__u42___u40_3_u47_3_u41_():
    return (round(2.0) * round(2.0))

@fpy(
    meta={
        'name': 'Test / (1/3)',
        'spec': lambda : round(1.0),
    }
)
def Test__u47___u40_1_u47_3_u41_():
    return (round(1.0) / round(1.0))

@fpy(
    meta={
        'name': 'Test / (2/3)',
        'spec': lambda : round(-0.5),
    }
)
def Test__u47___u40_2_u47_3_u41_():
    return (round(-1.0) / round(2.0))

@fpy(
    meta={
        'name': 'Test / (3/3)',
        'spec': lambda : round(-1.0),
    }
)
def Test__u47___u40_3_u47_3_u41_():
    return (round(2.0) / round(-2.0))

@fpy(
    meta={
        'name': 'Test fabs (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_fabs__u40_1_u47_3_u41_():
    return abs(round(0.0))

@fpy(
    meta={
        'name': 'Test fabs (2/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_fabs__u40_2_u47_3_u41_():
    return abs(round(-1.0))

@fpy(
    meta={
        'name': 'Test fabs (3/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_fabs__u40_3_u47_3_u41_():
    return abs(round(1.0))

@fpy(
    meta={
        'name': 'Test fma (1/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_fma__u40_1_u47_5_u41_():
    return fma(round(0.0), round(1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test fma (2/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fma__u40_2_u47_5_u41_():
    return fma(round(1.0), round(1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test fma (3/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_fma__u40_3_u47_5_u41_():
    return fma(round(1.0), round(-1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test fma (4/5)',
        'spec': lambda : round(2.0),
    }
)
def Test_fma__u40_4_u47_5_u41_():
    return fma(round(1.0), round(1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test fma (5/5)',
        'spec': lambda : round(-2.0),
    }
)
def Test_fma__u40_5_u47_5_u41_():
    return fma(round(-1.0), round(2.0), round(0.0))

@fpy(
    meta={
        'name': 'Test exp (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_exp__u40_1_u47_1_u41_():
    return exp(round(0.0))

@fpy(
    meta={
        'name': 'Test exp2 (1/4)',
        'spec': lambda : round(0.5),
    }
)
def Test_exp2__u40_1_u47_4_u41_():
    return exp2(round(-1.0))

@fpy(
    meta={
        'name': 'Test exp2 (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_exp2__u40_2_u47_4_u41_():
    return exp2(round(0.0))

@fpy(
    meta={
        'name': 'Test exp2 (3/4)',
        'spec': lambda : round(2.0),
    }
)
def Test_exp2__u40_3_u47_4_u41_():
    return exp2(round(1.0))

@fpy(
    meta={
        'name': 'Test exp2 (4/4)',
        'spec': lambda : round(4.0),
    }
)
def Test_exp2__u40_4_u47_4_u41_():
    return exp2(round(2.0))

@fpy(
    meta={
        'name': 'Test expm1 (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_expm1__u40_1_u47_1_u41_():
    return expm1(round(0.0))

@fpy(
    meta={
        'name': 'Test log (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_log__u40_1_u47_1_u41_():
    return log(round(1.0))

@fpy(
    meta={
        'name': 'Test log10 (1/2)',
        'spec': lambda : round(0.0),
    }
)
def Test_log10__u40_1_u47_2_u41_():
    return log10(round(1.0))

@fpy(
    meta={
        'name': 'Test log10 (2/2)',
        'spec': lambda : round(1.0),
    }
)
def Test_log10__u40_2_u47_2_u41_():
    return log10(round(10.0))

@fpy(
    meta={
        'name': 'Test log2 (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_log2__u40_1_u47_3_u41_():
    return log2(round(1.0))

@fpy(
    meta={
        'name': 'Test log2 (2/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_log2__u40_2_u47_3_u41_():
    return log2(round(2.0))

@fpy(
    meta={
        'name': 'Test log2 (3/3)',
        'spec': lambda : round(2.0),
    }
)
def Test_log2__u40_3_u47_3_u41_():
    return log2(round(4.0))

@fpy(
    meta={
        'name': 'Test log1p (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_log1p__u40_1_u47_1_u41_():
    return log1p(round(0.0))

@fpy(
    meta={
        'name': 'Test pow (1/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_pow__u40_1_u47_5_u41_():
    return pow(round(0.0), round(1.0))

@fpy(
    meta={
        'name': 'Test pow (2/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_pow__u40_2_u47_5_u41_():
    return pow(round(1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test pow (3/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_pow__u40_3_u47_5_u41_():
    return pow(round(1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test pow (4/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_pow__u40_4_u47_5_u41_():
    return pow(round(1.0), round(2.0))

@fpy(
    meta={
        'name': 'Test pow (5/5)',
        'spec': lambda : round(4.0),
    }
)
def Test_pow__u40_5_u47_5_u41_():
    return pow(round(2.0), round(2.0))

@fpy(
    meta={
        'name': 'Test sqrt (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_sqrt__u40_1_u47_3_u41_():
    return sqrt(round(0.0))

@fpy(
    meta={
        'name': 'Test sqrt (2/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_sqrt__u40_2_u47_3_u41_():
    return sqrt(round(1.0))

@fpy(
    meta={
        'name': 'Test sqrt (3/3)',
        'spec': lambda : round(2.0),
    }
)
def Test_sqrt__u40_3_u47_3_u41_():
    return sqrt(round(4.0))

@fpy(
    meta={
        'name': 'Test cbrt (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_cbrt__u40_1_u47_3_u41_():
    return cbrt(round(0.0))

@fpy(
    meta={
        'name': 'Test cbrt (2/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_cbrt__u40_2_u47_3_u41_():
    return cbrt(round(1.0))

@fpy(
    meta={
        'name': 'Test cbrt (3/3)',
        'spec': lambda : round(2.0),
    }
)
def Test_cbrt__u40_3_u47_3_u41_():
    return cbrt(round(8.0))

@fpy(
    meta={
        'name': 'Test hypot (1/2)',
        'spec': lambda : round(0.0),
    }
)
def Test_hypot__u40_1_u47_2_u41_():
    return hypot(round(0.0), round(0.0))

@fpy(
    meta={
        'name': 'Test hypot (2/2)',
        'spec': lambda : round(5.0),
    }
)
def Test_hypot__u40_2_u47_2_u41_():
    return hypot(round(3.0), round(4.0))

@fpy(
    meta={
        'name': 'Test sin (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_sin__u40_1_u47_1_u41_():
    return sin(round(0.0))

@fpy(
    meta={
        'name': 'Test cos (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_cos__u40_1_u47_1_u41_():
    return cos(round(0.0))

@fpy(
    meta={
        'name': 'Test tan (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_tan__u40_1_u47_1_u41_():
    return tan(round(0.0))

@fpy(
    meta={
        'name': 'Test asin (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_asin__u40_1_u47_1_u41_():
    return asin(round(0.0))

@fpy(
    meta={
        'name': 'Test acos (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_acos__u40_1_u47_1_u41_():
    return acos(round(1.0))

@fpy(
    meta={
        'name': 'Test atan (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_atan__u40_1_u47_1_u41_():
    return atan(round(0.0))

@fpy(
    meta={
        'name': 'Test atan2 (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_atan2__u40_1_u47_1_u41_():
    return atan2(round(0.0), round(1.0))

@fpy(
    meta={
        'name': 'Test sinh (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_sinh__u40_1_u47_1_u41_():
    return sinh(round(0.0))

@fpy(
    meta={
        'name': 'Test cosh (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_cosh__u40_1_u47_1_u41_():
    return cosh(round(0.0))

@fpy(
    meta={
        'name': 'Test tanh (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_tanh__u40_1_u47_1_u41_():
    return tanh(round(0.0))

@fpy(
    meta={
        'name': 'Test asinh (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_asinh__u40_1_u47_1_u41_():
    return asinh(round(0.0))

@fpy(
    meta={
        'name': 'Test acosh (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_acosh__u40_1_u47_1_u41_():
    return acosh(round(1.0))

@fpy(
    meta={
        'name': 'Test atanh (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_atanh__u40_1_u47_1_u41_():
    return atanh(round(0.0))

@fpy(
    meta={
        'name': 'Test erf (1/1)',
        'spec': lambda : round(0.0),
    }
)
def Test_erf__u40_1_u47_1_u41_():
    return erf(round(0.0))

@fpy(
    meta={
        'name': 'Test erfc (1/1)',
        'spec': lambda : round(1.0),
    }
)
def Test_erfc__u40_1_u47_1_u41_():
    return erfc(round(0.0))

@fpy(
    meta={
        'name': 'Test tgamma (1/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_tgamma__u40_1_u47_3_u41_():
    return tgamma(round(1.0))

@fpy(
    meta={
        'name': 'Test tgamma (2/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_tgamma__u40_2_u47_3_u41_():
    return tgamma(round(2.0))

@fpy(
    meta={
        'name': 'Test tgamma (3/3)',
        'spec': lambda : round(2.0),
    }
)
def Test_tgamma__u40_3_u47_3_u41_():
    return tgamma(round(3.0))

@fpy(
    meta={
        'name': 'Test lgamma (1/2)',
        'spec': lambda : round(0.0),
    }
)
def Test_lgamma__u40_1_u47_2_u41_():
    return lgamma(round(1.0))

@fpy(
    meta={
        'name': 'Test lgamma (2/2)',
        'spec': lambda : round(0.0),
    }
)
def Test_lgamma__u40_2_u47_2_u41_():
    return lgamma(round(2.0))

@fpy(
    meta={
        'name': 'Test ceil (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_ceil__u40_1_u47_4_u41_():
    return ceil(round(0.0))

@fpy(
    meta={
        'name': 'Test ceil (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_ceil__u40_2_u47_4_u41_():
    return ceil(round(0.25))

@fpy(
    meta={
        'name': 'Test ceil (3/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_ceil__u40_3_u47_4_u41_():
    return ceil(round(0.75))

@fpy(
    meta={
        'name': 'Test ceil (4/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_ceil__u40_4_u47_4_u41_():
    return ceil(round(1.0))

@fpy(
    meta={
        'name': 'Test floor (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_floor__u40_1_u47_4_u41_():
    return floor(round(0.0))

@fpy(
    meta={
        'name': 'Test floor (2/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_floor__u40_2_u47_4_u41_():
    return floor(round(0.25))

@fpy(
    meta={
        'name': 'Test floor (3/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_floor__u40_3_u47_4_u41_():
    return floor(round(0.75))

@fpy(
    meta={
        'name': 'Test floor (4/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_floor__u40_4_u47_4_u41_():
    return floor(round(1.0))

@fpy(
    meta={
        'name': 'Test fmod (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_fmod__u40_1_u47_3_u41_():
    return fmod(round(1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test fmod (2/3)',
        'spec': lambda : round(0.25),
    }
)
def Test_fmod__u40_2_u47_3_u41_():
    return fmod(round(1.25), round(1.0))

@fpy(
    meta={
        'name': 'Test fmod (3/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmod__u40_3_u47_3_u41_():
    return fmod(round(3.0), round(2.0))

@fpy(
    meta={
        'name': 'Test remainder (1/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_remainder__u40_1_u47_3_u41_():
    return remainder(round(1), round(1))

@fpy(
    meta={
        'name': 'Test remainder (2/3)',
        'spec': lambda : round(0.25),
    }
)
def Test_remainder__u40_2_u47_3_u41_():
    return remainder(round(1.25), round(1))

@fpy(
    meta={
        'name': 'Test remainder (3/3)',
        'spec': lambda : round(-1.0),
    }
)
def Test_remainder__u40_3_u47_3_u41_():
    return remainder(round(3), round(2))

@fpy(
    meta={
        'name': 'Test fmax (1/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmax__u40_1_u47_5_u41_():
    return max(round(1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test fmax (2/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_fmax__u40_2_u47_5_u41_():
    return max(round(-1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test fmax (3/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmax__u40_3_u47_5_u41_():
    return max(round(1.0), round(-1.0))

@fpy(
    meta={
        'name': 'Test fmax (4/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmax__u40_4_u47_5_u41_():
    return max(round(1.0), nan())

@fpy(
    meta={
        'name': 'Test fmax (5/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmax__u40_5_u47_5_u41_():
    return max(nan(), round(1.0))

@fpy(
    meta={
        'name': 'Test fmin (1/5)',
        'spec': lambda : round(0.0),
    }
)
def Test_fmin__u40_1_u47_5_u41_():
    return min(round(1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test fmin (2/5)',
        'spec': lambda : round(-1.0),
    }
)
def Test_fmin__u40_2_u47_5_u41_():
    return min(round(-1.0), round(0.0))

@fpy(
    meta={
        'name': 'Test fmin (3/5)',
        'spec': lambda : round(-1.0),
    }
)
def Test_fmin__u40_3_u47_5_u41_():
    return min(round(1.0), round(-1.0))

@fpy(
    meta={
        'name': 'Test fmin (4/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmin__u40_4_u47_5_u41_():
    return min(round(1.0), nan())

@fpy(
    meta={
        'name': 'Test fmin (5/5)',
        'spec': lambda : round(1.0),
    }
)
def Test_fmin__u40_5_u47_5_u41_():
    return min(nan(), round(1.0))

@fpy(
    meta={
        'name': 'Test fdim (1/3)',
        'spec': lambda : round(1.0),
    }
)
def Test_fdim__u40_1_u47_3_u41_():
    return fdim(round(2.0), round(1.0))

@fpy(
    meta={
        'name': 'Test fdim (2/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_fdim__u40_2_u47_3_u41_():
    return fdim(round(1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test fdim (3/3)',
        'spec': lambda : round(0.0),
    }
)
def Test_fdim__u40_3_u47_3_u41_():
    return fdim(round(1.0), round(2.0))

@fpy(
    meta={
        'name': 'Test copysign (1/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_copysign__u40_1_u47_4_u41_():
    return copysign(round(1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test copysign (2/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_copysign__u40_2_u47_4_u41_():
    return copysign(round(-1.0), round(1.0))

@fpy(
    meta={
        'name': 'Test copysign (3/4)',
        'spec': lambda : round(-1.0),
    }
)
def Test_copysign__u40_3_u47_4_u41_():
    return copysign(round(1.0), round(-1.0))

@fpy(
    meta={
        'name': 'Test copysign (4/4)',
        'spec': lambda : round(-1.0),
    }
)
def Test_copysign__u40_4_u47_4_u41_():
    return copysign(round(-1.0), round(-1.0))

@fpy(
    meta={
        'name': 'Test trunc (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_trunc__u40_1_u47_4_u41_():
    return trunc(round(0.0))

@fpy(
    meta={
        'name': 'Test trunc (2/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_trunc__u40_2_u47_4_u41_():
    return trunc(round(0.25))

@fpy(
    meta={
        'name': 'Test trunc (3/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_trunc__u40_3_u47_4_u41_():
    return trunc(round(-0.75))

@fpy(
    meta={
        'name': 'Test trunc (4/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_trunc__u40_4_u47_4_u41_():
    return trunc(round(1.0))

@fpy(
    meta={
        'name': 'Test round (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_round__u40_1_u47_4_u41_():
    return round(round(0.0))

@fpy(
    meta={
        'name': 'Test round (2/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_round__u40_2_u47_4_u41_():
    return round(round(0.25))

@fpy(
    meta={
        'name': 'Test round (3/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_round__u40_3_u47_4_u41_():
    return round(round(0.75))

@fpy(
    meta={
        'name': 'Test round (4/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_round__u40_4_u47_4_u41_():
    return round(round(1.0))

@fpy(
    meta={
        'name': 'Test nearbyint (1/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_nearbyint__u40_1_u47_4_u41_():
    return nearbyint(round(0.0))

@fpy(
    meta={
        'name': 'Test nearbyint (2/4)',
        'spec': lambda : round(0.0),
    }
)
def Test_nearbyint__u40_2_u47_4_u41_():
    return nearbyint(round(0.25))

@fpy(
    meta={
        'name': 'Test nearbyint (3/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_nearbyint__u40_3_u47_4_u41_():
    return nearbyint(round(0.75))

@fpy(
    meta={
        'name': 'Test nearbyint (4/4)',
        'spec': lambda : round(1.0),
    }
)
def Test_nearbyint__u40_4_u47_4_u41_():
    return nearbyint(round(1.0))

