from fpy2 import *
from fpy2.typing import *

@fpy(
    name='Test < (1/9)',
    spec=lambda : 0.0,
)
def Test__u60___u40_1_u47_9_u41_():
    if 1.0 < 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (2/9)',
    spec=lambda : 1.0,
)
def Test__u60___u40_2_u47_9_u41_():
    if 0.0 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (3/9)',
    spec=lambda : 1.0,
)
def Test__u60___u40_3_u47_9_u41_():
    if -1.0 < 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (4/9)',
    spec=lambda : 0.0,
)
def Test__u60___u40_4_u47_9_u41_():
    if 0.0 < -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (5/9)',
    spec=lambda : 0.0,
)
def Test__u60___u40_5_u47_9_u41_():
    if 1.0 < -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (6/9)',
    spec=lambda : 1.0,
)
def Test__u60___u40_6_u47_9_u41_():
    if -1.0 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (7/9)',
    spec=lambda : 0.0,
)
def Test__u60___u40_7_u47_9_u41_():
    if 0.0 < 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (8/9)',
    spec=lambda : 0.0,
)
def Test__u60___u40_8_u47_9_u41_():
    if NAN < NAN:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test < (9/9)',
    spec=lambda : 0.0,
)
def Test__u60___u40_9_u47_9_u41_():
    if 0.0 < 0.0 < 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (1/9)',
    spec=lambda : 1.0,
)
def Test__u62___u40_1_u47_9_u41_():
    if 1.0 > 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (2/9)',
    spec=lambda : 0.0,
)
def Test__u62___u40_2_u47_9_u41_():
    if 0.0 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (3/9)',
    spec=lambda : 0.0,
)
def Test__u62___u40_3_u47_9_u41_():
    if -1.0 > 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (4/9)',
    spec=lambda : 1.0,
)
def Test__u62___u40_4_u47_9_u41_():
    if 0.0 > -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (5/9)',
    spec=lambda : 1.0,
)
def Test__u62___u40_5_u47_9_u41_():
    if 1.0 > -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (6/9)',
    spec=lambda : 0.0,
)
def Test__u62___u40_6_u47_9_u41_():
    if -1.0 > 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (7/9)',
    spec=lambda : 0.0,
)
def Test__u62___u40_7_u47_9_u41_():
    if 0.0 > 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (8/9)',
    spec=lambda : 0.0,
)
def Test__u62___u40_8_u47_9_u41_():
    if NAN > NAN:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test > (9/9)',
    spec=lambda : 0.0,
)
def Test__u62___u40_9_u47_9_u41_():
    if 0.0 > 0.0 > 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (1/9)',
    spec=lambda : 0.0,
)
def Test__u60__u61___u40_1_u47_9_u41_():
    if 1.0 <= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (2/9)',
    spec=lambda : 1.0,
)
def Test__u60__u61___u40_2_u47_9_u41_():
    if 0.0 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (3/9)',
    spec=lambda : 1.0,
)
def Test__u60__u61___u40_3_u47_9_u41_():
    if -1.0 <= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (4/9)',
    spec=lambda : 0.0,
)
def Test__u60__u61___u40_4_u47_9_u41_():
    if 0.0 <= -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (5/9)',
    spec=lambda : 0.0,
)
def Test__u60__u61___u40_5_u47_9_u41_():
    if 1.0 <= -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (6/9)',
    spec=lambda : 1.0,
)
def Test__u60__u61___u40_6_u47_9_u41_():
    if -1.0 <= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (7/9)',
    spec=lambda : 1.0,
)
def Test__u60__u61___u40_7_u47_9_u41_():
    if 0.0 <= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (8/9)',
    spec=lambda : 0.0,
)
def Test__u60__u61___u40_8_u47_9_u41_():
    if NAN <= NAN:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test <= (9/9)',
    spec=lambda : 1.0,
)
def Test__u60__u61___u40_9_u47_9_u41_():
    if 0.0 <= 0.0 <= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (1/9)',
    spec=lambda : 1.0,
)
def Test__u62__u61___u40_1_u47_9_u41_():
    if 1.0 >= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (2/9)',
    spec=lambda : 0.0,
)
def Test__u62__u61___u40_2_u47_9_u41_():
    if 0.0 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (3/9)',
    spec=lambda : 0.0,
)
def Test__u62__u61___u40_3_u47_9_u41_():
    if -1.0 >= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (4/9)',
    spec=lambda : 1.0,
)
def Test__u62__u61___u40_4_u47_9_u41_():
    if 0.0 >= -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (5/9)',
    spec=lambda : 1.0,
)
def Test__u62__u61___u40_5_u47_9_u41_():
    if 1.0 >= -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (6/9)',
    spec=lambda : 0.0,
)
def Test__u62__u61___u40_6_u47_9_u41_():
    if -1.0 >= 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (7/9)',
    spec=lambda : 1.0,
)
def Test__u62__u61___u40_7_u47_9_u41_():
    if 0.0 >= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (8/9)',
    spec=lambda : 0.0,
)
def Test__u62__u61___u40_8_u47_9_u41_():
    if NAN >= NAN:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test >= (9/9)',
    spec=lambda : 1.0,
)
def Test__u62__u61___u40_9_u47_9_u41_():
    if 0.0 >= 0.0 >= 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (1/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_1_u47_9_u41_():
    if 1.0 == 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (2/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_2_u47_9_u41_():
    if 0.0 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (3/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_3_u47_9_u41_():
    if -1.0 == 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (4/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_4_u47_9_u41_():
    if 0.0 == -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (5/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_5_u47_9_u41_():
    if 1.0 == -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (6/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_6_u47_9_u41_():
    if -1.0 == 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (7/9)',
    spec=lambda : 1.0,
)
def Test__u61__u61___u40_7_u47_9_u41_():
    if 0.0 == 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (8/9)',
    spec=lambda : 0.0,
)
def Test__u61__u61___u40_8_u47_9_u41_():
    if NAN == NAN:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test == (9/9)',
    spec=lambda : 1.0,
)
def Test__u61__u61___u40_9_u47_9_u41_():
    if 0.0 == 0.0 == 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (1/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_1_u47_9_u41_():
    if 1.0 != 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (2/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_2_u47_9_u41_():
    if 0.0 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (3/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_3_u47_9_u41_():
    if -1.0 != 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (4/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_4_u47_9_u41_():
    if 0.0 != -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (5/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_5_u47_9_u41_():
    if 1.0 != -1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (6/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_6_u47_9_u41_():
    if -1.0 != 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (7/9)',
    spec=lambda : 0.0,
)
def Test__u33__u61___u40_7_u47_9_u41_():
    if 0.0 != 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (8/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_8_u47_9_u41_():
    if NAN != NAN:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test != (9/9)',
    spec=lambda : 1.0,
)
def Test__u33__u61___u40_9_u47_9_u41_():
    if 0.0 == 0.0 == 0.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test and (1/6)',
    spec=lambda : 1.0,
)
def Test_and__u40_1_u47_6_u41_():
    if True and True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test and (2/6)',
    spec=lambda : 0.0,
)
def Test_and__u40_2_u47_6_u41_():
    if True and False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test and (3/6)',
    spec=lambda : 0.0,
)
def Test_and__u40_3_u47_6_u41_():
    if False and True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test and (4/6)',
    spec=lambda : 0.0,
)
def Test_and__u40_4_u47_6_u41_():
    if False and False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test and (5/6)',
    spec=lambda : 1.0,
)
def Test_and__u40_5_u47_6_u41_():
    if True and True and True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test and (6/6)',
    spec=lambda : 0.0,
)
def Test_and__u40_6_u47_6_u41_():
    if True and True and False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test or (1/4)',
    spec=lambda : 1.0,
)
def Test_or__u40_1_u47_4_u41_():
    if True or True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test or (2/4)',
    spec=lambda : 1.0,
)
def Test_or__u40_2_u47_4_u41_():
    if True or False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test or (3/4)',
    spec=lambda : 1.0,
)
def Test_or__u40_3_u47_4_u41_():
    if False or True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test or (4/4)',
    spec=lambda : 0.0,
)
def Test_or__u40_4_u47_4_u41_():
    if False or False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test or (5/6)',
    spec=lambda : 0.0,
)
def Test_or__u40_5_u47_6_u41_():
    if False or False or False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test or (6/6)',
    spec=lambda : 1.0,
)
def Test_or__u40_6_u47_6_u41_():
    if False or False or True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test not (1/2)',
    spec=lambda : 0.0,
)
def Test_not__u40_1_u47_2_u41_():
    if not True:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test not (2/2)',
    spec=lambda : 1.0,
)
def Test_not__u40_2_u47_2_u41_():
    if not False:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isinf (1/4)',
    spec=lambda : 0.0,
)
def Test_isinf__u40_1_u47_4_u41_():
    if isinf(0.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isinf (2/4)',
    spec=lambda : 0.0,
)
def Test_isinf__u40_2_u47_4_u41_():
    if isinf(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isinf (3/4)',
    spec=lambda : 1.0,
)
def Test_isinf__u40_3_u47_4_u41_():
    if isinf(INFINITY):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isinf (4/4)',
    spec=lambda : 0.0,
)
def Test_isinf__u40_4_u47_4_u41_():
    if isinf(NAN):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnan (1/4)',
    spec=lambda : 0.0,
)
def Test_isnan__u40_1_u47_4_u41_():
    if isnan(0.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnan (2/4)',
    spec=lambda : 0.0,
)
def Test_isnan__u40_2_u47_4_u41_():
    if isnan(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnan (3/4)',
    spec=lambda : 0.0,
)
def Test_isnan__u40_3_u47_4_u41_():
    if isnan(INFINITY):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnan (4/4)',
    spec=lambda : 1.0,
)
def Test_isnan__u40_4_u47_4_u41_():
    if isnan(NAN):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isfinite (1/4)',
    spec=lambda : 1.0,
)
def Test_isfinite__u40_1_u47_4_u41_():
    if isfinite(0.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isfinite (2/4)',
    spec=lambda : 1.0,
)
def Test_isfinite__u40_2_u47_4_u41_():
    if isfinite(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isfinite (3/4)',
    spec=lambda : 0.0,
)
def Test_isfinite__u40_3_u47_4_u41_():
    if isfinite(INFINITY):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isfinite (4/4)',
    spec=lambda : 0.0,
)
def Test_isfinite__u40_4_u47_4_u41_():
    if isfinite(NAN):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnormal (1/4)',
    spec=lambda : 0.0,
)
def Test_isnormal__u40_1_u47_4_u41_():
    if isnormal(0.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnormal (2/4)',
    spec=lambda : 1.0,
)
def Test_isnormal__u40_2_u47_4_u41_():
    if isnormal(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnormal (3/4)',
    spec=lambda : 0.0,
)
def Test_isnormal__u40_3_u47_4_u41_():
    if isnormal(INFINITY):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test isnormal (4/4)',
    spec=lambda : 0.0,
)
def Test_isnormal__u40_4_u47_4_u41_():
    if isnormal(NAN):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test signbit (1/3)',
    spec=lambda : 0.0,
)
def Test_signbit__u40_1_u47_3_u41_():
    if signbit(0.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test signbit (2/3)',
    spec=lambda : 0.0,
)
def Test_signbit__u40_2_u47_3_u41_():
    if signbit(1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test signbit (3/3)',
    spec=lambda : 1.0,
)
def Test_signbit__u40_3_u47_3_u41_():
    if signbit(-1.0):
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test E (1/1)',
    spec=lambda : 1.0,
)
def Test_E__u40_1_u47_1_u41_():
    if 2.0 < E and E < 3.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test LOG2E (1/1)',
    spec=lambda : 1.0,
)
def Test_LOG2E__u40_1_u47_1_u41_():
    if 1.0 < LOG2E and LOG2E < 2.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test LOG10E (1/1)',
    spec=lambda : 1.0,
)
def Test_LOG10E__u40_1_u47_1_u41_():
    if 0.25 < LOG10E and LOG10E < 0.5:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test LN2 (1/1)',
    spec=lambda : 1.0,
)
def Test_LN2__u40_1_u47_1_u41_():
    if 0.5 < LN2 and LN2 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test PI (1/1)',
    spec=lambda : 1.0,
)
def Test_PI__u40_1_u47_1_u41_():
    if 3.0 < PI and PI < 4.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test PI_2 (1/1)',
    spec=lambda : 1.0,
)
def Test_PI_2__u40_1_u47_1_u41_():
    if 1.0 < PI_2 and PI_2 < 2.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test PI_4 (1/1)',
    spec=lambda : 1.0,
)
def Test_PI_4__u40_1_u47_1_u41_():
    if 0.5 < PI_4 and PI_4 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test M_1_PI (1/1)',
    spec=lambda : 1.0,
)
def Test_M_1_PI__u40_1_u47_1_u41_():
    if 0.25 < M_1_PI and M_1_PI < 0.5:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test M_2_PI (1/1)',
    spec=lambda : 1.0,
)
def Test_M_2_PI__u40_1_u47_1_u41_():
    if 0.5 < M_2_PI and M_2_PI < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test M_2_SQRTPI (1/1)',
    spec=lambda : 1.0,
)
def Test_M_2_SQRTPI__u40_1_u47_1_u41_():
    if 1.0 < M_2_SQRTPI and M_2_SQRTPI < 2.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test SQRT2 (1/1)',
    spec=lambda : 1.0,
)
def Test_SQRT2__u40_1_u47_1_u41_():
    if 1.0 < SQRT2 and SQRT2 < 2.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test SQRT1_2 (1/1)',
    spec=lambda : 1.0,
)
def Test_SQRT1_2__u40_1_u47_1_u41_():
    if 0.5 < SQRT1_2 and SQRT1_2 < 1.0:
        t = 1
    else:
        t = 0
    return t

@fpy(
    name='Test let (1/5)',
    spec=lambda : 1.0,
)
def Test_let__u40_1_u47_5_u41_():
    a = 1.0
    return a

@fpy(
    name='Test let (2/5)',
    spec=lambda : 2.0,
)
def Test_let__u40_2_u47_5_u41_():
    a = 1.0
    b = 1.0
    return (a + b)

@fpy(
    name='Test let (3/5)',
    spec=lambda : -1.0,
)
def Test_let__u40_3_u47_5_u41_():
    a = 1.0
    b = 0.0
    c = -1.0
    d = 0
    if a > b:
        t = c
    else:
        t = d
    return t

@fpy(
    name='Test let (4/5)',
    spec=lambda : 1.0,
)
def Test_let__u40_4_u47_5_u41_():
    a = 1.0
    b = 0.0
    a0 = b
    b1 = a
    return b1

@fpy(
    name='Test let (5/5)',
    spec=lambda : 0.0,
)
def Test_let__u40_5_u47_5_u41_():
    a = 1.0
    b = 0.0
    a0 = b
    b1 = a
    a2 = b1
    b3 = a0
    return b3

@fpy(
    name='Test let* (1/5)',
    spec=lambda : 1.0,
)
def Test_let_u42___u40_1_u47_5_u41_():
    a = 1.0
    return a

@fpy(
    name='Test let* (2/5)',
    spec=lambda : 0.0,
)
def Test_let_u42___u40_2_u47_5_u41_():
    a = 1.0
    b = -1.0
    return (a + b)

@fpy(
    name='Test let* (3/5)',
    spec=lambda : -1.0,
)
def Test_let_u42___u40_3_u47_5_u41_():
    a = 1.0
    b = 0.0
    c = -1.0
    d = 0
    if a > b:
        t = c
    else:
        t = d
    return t

@fpy(
    name='Test let* (4/5)',
    spec=lambda : 1.0,
)
def Test_let_u42___u40_4_u47_5_u41_():
    a = 1.0
    b = a
    return b

@fpy(
    name='Test let* (5/5)',
    spec=lambda : 5.0,
)
def Test_let_u42___u40_5_u47_5_u41_():
    a = 3.0
    b = 1.0
    c = -1.0
    b0 = a
    c1 = (c + (b0 + a))
    a2 = c1
    b3 = a2
    return b3

@fpy(
    name='Test while (1/7)',
    spec=lambda : 0.0,
)
def Test_while__u40_1_u47_7_u41_():
    a = 0.0
    while False:
        t = (a + 1.0)
        a = t
    return a

@fpy(
    name='Test while (2/7)',
    spec=lambda : 4.0,
)
def Test_while__u40_2_u47_7_u41_():
    a = 0.0
    while a < 4:
        t = (a + 1.0)
        a = t
    return a

@fpy(
    name='Test while (3/7)',
    spec=lambda : 8.0,
)
def Test_while__u40_3_u47_7_u41_():
    a = 0.0
    b = 1.0
    while a < 3:
        t = (a + 1.0)
        t0 = (b * 2.0)
        a = t
        b = t0
    return b

@fpy(
    name='Test while (4/7)',
    spec=lambda : -6.0,
)
def Test_while__u40_4_u47_7_u41_():
    a = 0.0
    b = 0.0
    while a <= 3:
        t = (a + 1.0)
        t0 = (b - a)
        a = t
        b = t0
    return b

@fpy(
    name='Test while (5/7)',
    spec=lambda : 1.0,
)
def Test_while__u40_5_u47_7_u41_():
    i = 0
    a = 0.0
    b = 1.0
    while i < 3:
        t = (i + 1)
        t0 = b
        t1 = a
        i = t
        a = t0
        b = t1
    return a

@fpy(
    name='Test while (6/7)',
    spec=lambda : 0.0,
)
def Test_while__u40_6_u47_7_u41_():
    i = 0
    a = 0.0
    b = 1.0
    while i < 4:
        t = (i + 1)
        t0 = b
        t1 = a
        i = t
        a = t0
        b = t1
    return a

@fpy(
    name='Test while (7/7)',
    spec=lambda : 6.0,
)
def Test_while__u40_7_u47_7_u41_():
    a = 0.0
    b = 0.0
    while a <= 3:
        t = (a + 1.0)
        i = 0
        x = 0.0
        while i <= a:
            t0 = (i + 1)
            t1 = (x + i)
            i = t0
            x = t1
        t2 = x
        a = t
        b = t2
    return b

@fpy(
    name='Test while* (1/7)',
    spec=lambda : 0.0,
)
def Test_while_u42___u40_1_u47_7_u41_():
    a = 0.0
    while False:
        a = (a + 1.0)
    return a

@fpy(
    name='Test while* (2/7)',
    spec=lambda : 4.0,
)
def Test_while_u42___u40_2_u47_7_u41_():
    a = 0.0
    while a < 4:
        a = (a + 1.0)
    return a

@fpy(
    name='Test while* (3/7)',
    spec=lambda : 8.0,
)
def Test_while_u42___u40_3_u47_7_u41_():
    a = 0.0
    b = 1.0
    while a < 3:
        a = (a + 1.0)
        b = (b * 2.0)
    return b

@fpy(
    name='Test while* (4/7)',
    spec=lambda : -10.0,
)
def Test_while_u42___u40_4_u47_7_u41_():
    a = 0.0
    b = 0.0
    while a <= 3:
        a = (a + 1.0)
        b = (b - a)
    return b

@fpy(
    name='Test while* (5/7)',
    spec=lambda : 1.0,
)
def Test_while_u42___u40_5_u47_7_u41_():
    i = 0.0
    a = 0.0
    b = 1.0
    while i < 3:
        i = (i + 1)
        a = b
        b = a
    return a

@fpy(
    name='Test while* (6/7)',
    spec=lambda : 1.0,
)
def Test_while_u42___u40_6_u47_7_u41_():
    i = 0.0
    a = 0.0
    b = 1.0
    while i < 4:
        i = (i + 1)
        a = b
        b = a
    return a

@fpy(
    name='Test while* (7/7)',
    spec=lambda : 15.0,
)
def Test_while_u42___u40_7_u47_7_u41_():
    a = 0.0
    b = 0.0
    while a <= 3:
        a = (a + 1.0)
        i = 0
        x = 0.0
        while i <= a:
            i = (i + 1)
            x = (x + i)
        b = x
    return b

@fpy(
    name='Test if (1/6)',
    spec=lambda : 1.0,
)
def Test_if__u40_1_u47_6_u41_():
    if 1.0 > 0.0:
        t = 1.0
    else:
        t = 0.0
    return t

@fpy(
    name='Test if (2/6)',
    spec=lambda : 1.0,
)
def Test_if__u40_2_u47_6_u41_():
    if 1.0 < 0.0:
        t = 0.0
    else:
        t = 1.0
    return t

@fpy(name='Test if (3/6)')
def Test_if__u40_3_u47_6_u41_():
    if True:
        y = 1
        t2 = y
    else:
        if False:
            y0 = 2
            t = y0
        else:
            y1 = 3
            t = y1
        t2 = t
    return t2

@fpy(
    name='Test if (4/6)',
    spec=lambda : 1.0,
)
def Test_if__u40_4_u47_6_u41_():
    if True:
        t1 = 1
    else:
        t = False
        if t:
            t0 = 0
        else:
            t0 = 0
        t1 = t0
    return t1

@fpy(name='Test if (5/6)')
def Test_if__u40_5_u47_6_u41_():
    t = False
    if t:
        t4 = 0
    else:
        t0 = True
        if t0:
            t3 = 1
        else:
            t1 = True
            if t1:
                t2 = 0
            else:
                t2 = 0
            t3 = t2
        t4 = t3
    return t4

@fpy(
    name='Test if (6/6)',
    spec=lambda : 1.0,
)
def Test_if__u40_6_u47_6_u41_():
    if False:
        t1 = 1.0
    else:
        y = 1
        t = 2 > y
        if t:
            t0 = 1
        else:
            t0 = 0
        t1 = t0
    return t1

@fpy(
    name='Test cast (1/1)',
    spec=lambda : 1.0,
)
def Test_cast__u40_1_u47_1_u41_():
    return cast(1.0)

@fpy(
    name='Test ! (1/1)',
    spec=lambda : 1.0,
)
def Test__u33___u40_1_u47_1_u41_():
    with IEEEContext(es=11, nbits=64, rm=RoundingMode.RNE) as _:
        t = 1.0
    return t

@fpy(
    name='Test FPCore identifier (FPBench 2.0)',
    spec=lambda : 1.0,
)
def ident():
    return 1.0

@fpy(name='Test arguments (1/2)')
def Test_arguments__u40_1_u47_2_u41_(x):
    return x

@fpy(name='Test arguments (2/2)')
def Test_arguments__u40_2_u47_2_u41_(x, y):
    return x

@fpy(name='Nested syntax (1/2)')
def Nested_syntax__u40_1_u47_2_u41_(x):
    if x < 0:
        t = -1
    else:
        t = 1
    return (1 + t)

@fpy(name='Nested syntax (2/2)')
def Nested_syntax__u40_2_u47_2_u41_(x):
    y = 1
    return (1 + (y + x))

@fpy(
    name='Test decnum (1/5)',
    spec=lambda : 0.0,
)
def Test_decnum__u40_1_u47_5_u41_():
    return 0.0

@fpy(
    name='Test decnum (2/5)',
    spec=lambda : 1.0,
)
def Test_decnum__u40_2_u47_5_u41_():
    return 1.0

@fpy(
    name='Test decnum (3/5)',
    spec=lambda : -1.0,
)
def Test_decnum__u40_3_u47_5_u41_():
    return -1.0

@fpy(
    name='Test decnum (4/5)',
    spec=lambda : 1.5,
)
def Test_decnum__u40_4_u47_5_u41_():
    return 1.5

@fpy(
    name='Test decnum (5/5)',
    spec=lambda : 0.75,
)
def Test_decnum__u40_5_u47_5_u41_():
    return 0.75

@fpy(
    name='Test hexnum (1/6)',
    spec=lambda : 0.0,
)
def Test_hexnum__u40_1_u47_6_u41_():
    return hexfloat('0x0.0p+0')

@fpy(
    name='Test hexnum (2/6)',
    spec=lambda : 1.0,
)
def Test_hexnum__u40_2_u47_6_u41_():
    return hexfloat('0x1.0p+0')

@fpy(
    name='Test hexnum (3/6)',
    spec=lambda : -1.0,
)
def Test_hexnum__u40_3_u47_6_u41_():
    return hexfloat('-0x1.0p+0')

@fpy(
    name='Test hexnum (4/6)',
    spec=lambda : 0.5,
)
def Test_hexnum__u40_4_u47_6_u41_():
    return hexfloat('0x1.0p-1')

@fpy(
    name='Test hexnum (5/6)',
    spec=lambda : -2.0,
)
def Test_hexnum__u40_5_u47_6_u41_():
    return hexfloat('-0x1.0p+1')

@fpy(
    name='Test hexnum (6/6)',
    spec=lambda : 0.99609375,
)
def Test_hexnum__u40_6_u47_6_u41_():
    return hexfloat('0xf.fp-4')

@fpy(
    name='Test rational (1/5)',
    spec=lambda : 0.0,
)
def Test_rational__u40_1_u47_5_u41_():
    return 0

@fpy(
    name='Test rational (2/5)',
    spec=lambda : 1.0,
)
def Test_rational__u40_2_u47_5_u41_():
    return 1

@fpy(
    name='Test rational (3/5)',
    spec=lambda : -1.0,
)
def Test_rational__u40_3_u47_5_u41_():
    return -1

@fpy(
    name='Test rational (4/5)',
    spec=lambda : -0.5,
)
def Test_rational__u40_4_u47_5_u41_():
    return rational(-1, 2)

@fpy(
    name='Test rational (5/5)',
    spec=lambda : 1.25,
)
def Test_rational__u40_5_u47_5_u41_():
    return rational(5, 4)

@fpy(
    name='Test digits (1/4)',
    spec=lambda : 0.0,
)
def Test_digits__u40_1_u47_4_u41_():
    return digits(0, 0, 2)

@fpy(
    name='Test digits (2/4)',
    spec=lambda : 1.0,
)
def Test_digits__u40_2_u47_4_u41_():
    return digits(1, 0, 2)

@fpy(
    name='Test digits (3/4)',
    spec=lambda : -2.0,
)
def Test_digits__u40_3_u47_4_u41_():
    return digits(-1, 1, 2)

@fpy(
    name='Test digits (4/4)',
    spec=lambda : 1.5,
)
def Test_digits__u40_4_u47_4_u41_():
    return digits(3, -1, 2)

@fpy(
    name='Test - (1/2)',
    spec=lambda : -1.0,
)
def Test____u40_1_u47_2_u41_():
    return -1.0

@fpy(
    name='Test - (2/2)',
    spec=lambda : 1.0,
)
def Test____u40_2_u47_2_u41_():
    return --1.0

@fpy(
    name='Test + (1/3)',
    spec=lambda : 1.0,
)
def Test__u43___u40_1_u47_3_u41_():
    return (0.0 + 1.0)

@fpy(
    name='Test + (2/3)',
    spec=lambda : 2.0,
)
def Test__u43___u40_2_u47_3_u41_():
    return (1.0 + 1.0)

@fpy(
    name='Test + (3/3)',
    spec=lambda : 0.0,
)
def Test__u43___u40_3_u47_3_u41_():
    return (-1.0 + 1.0)

@fpy(
    name='Test - (1/3)',
    spec=lambda : -1.0,
)
def Test____u40_1_u47_3_u41_():
    return (0.0 - 1.0)

@fpy(
    name='Test - (2/3)',
    spec=lambda : 0.0,
)
def Test____u40_2_u47_3_u41_():
    return (1.0 - 1.0)

@fpy(
    name='Test - (3/3)',
    spec=lambda : -2.0,
)
def Test____u40_3_u47_3_u41_():
    return (-1.0 - 1.0)

@fpy(
    name='Test * (1/3)',
    spec=lambda : 0.0,
)
def Test__u42___u40_1_u47_3_u41_():
    return (0.0 * 1.0)

@fpy(
    name='Test * (2/3)',
    spec=lambda : -2.0,
)
def Test__u42___u40_2_u47_3_u41_():
    return (2.0 * -1.0)

@fpy(
    name='Test * (3/3)',
    spec=lambda : 4.0,
)
def Test__u42___u40_3_u47_3_u41_():
    return (2.0 * 2.0)

@fpy(
    name='Test / (1/3)',
    spec=lambda : 1.0,
)
def Test__u47___u40_1_u47_3_u41_():
    return (1.0 / 1.0)

@fpy(
    name='Test / (2/3)',
    spec=lambda : -0.5,
)
def Test__u47___u40_2_u47_3_u41_():
    return (-1.0 / 2.0)

@fpy(
    name='Test / (3/3)',
    spec=lambda : -1.0,
)
def Test__u47___u40_3_u47_3_u41_():
    return (2.0 / -2.0)

@fpy(
    name='Test fabs (1/3)',
    spec=lambda : 0.0,
)
def Test_fabs__u40_1_u47_3_u41_():
    return fabs(0.0)

@fpy(
    name='Test fabs (2/3)',
    spec=lambda : 1.0,
)
def Test_fabs__u40_2_u47_3_u41_():
    return fabs(-1.0)

@fpy(
    name='Test fabs (3/3)',
    spec=lambda : 1.0,
)
def Test_fabs__u40_3_u47_3_u41_():
    return fabs(1.0)

@fpy(
    name='Test fma (1/5)',
    spec=lambda : 0.0,
)
def Test_fma__u40_1_u47_5_u41_():
    return fma(0.0, 1.0, 0.0)

@fpy(
    name='Test fma (2/5)',
    spec=lambda : 1.0,
)
def Test_fma__u40_2_u47_5_u41_():
    return fma(1.0, 1.0, 0.0)

@fpy(
    name='Test fma (3/5)',
    spec=lambda : 0.0,
)
def Test_fma__u40_3_u47_5_u41_():
    return fma(1.0, -1.0, 1.0)

@fpy(
    name='Test fma (4/5)',
    spec=lambda : 2.0,
)
def Test_fma__u40_4_u47_5_u41_():
    return fma(1.0, 1.0, 1.0)

@fpy(
    name='Test fma (5/5)',
    spec=lambda : -2.0,
)
def Test_fma__u40_5_u47_5_u41_():
    return fma(-1.0, 2.0, 0.0)

@fpy(
    name='Test exp (1/1)',
    spec=lambda : 1.0,
)
def Test_exp__u40_1_u47_1_u41_():
    return exp(0.0)

@fpy(
    name='Test exp2 (1/4)',
    spec=lambda : 0.5,
)
def Test_exp2__u40_1_u47_4_u41_():
    return exp2(-1.0)

@fpy(
    name='Test exp2 (2/4)',
    spec=lambda : 1.0,
)
def Test_exp2__u40_2_u47_4_u41_():
    return exp2(0.0)

@fpy(
    name='Test exp2 (3/4)',
    spec=lambda : 2.0,
)
def Test_exp2__u40_3_u47_4_u41_():
    return exp2(1.0)

@fpy(
    name='Test exp2 (4/4)',
    spec=lambda : 4.0,
)
def Test_exp2__u40_4_u47_4_u41_():
    return exp2(2.0)

@fpy(
    name='Test expm1 (1/1)',
    spec=lambda : 0.0,
)
def Test_expm1__u40_1_u47_1_u41_():
    return expm1(0.0)

@fpy(
    name='Test log (1/1)',
    spec=lambda : 0.0,
)
def Test_log__u40_1_u47_1_u41_():
    return log(1.0)

@fpy(
    name='Test log10 (1/2)',
    spec=lambda : 0.0,
)
def Test_log10__u40_1_u47_2_u41_():
    return log10(1.0)

@fpy(
    name='Test log10 (2/2)',
    spec=lambda : 1.0,
)
def Test_log10__u40_2_u47_2_u41_():
    return log10(10.0)

@fpy(
    name='Test log2 (1/3)',
    spec=lambda : 0.0,
)
def Test_log2__u40_1_u47_3_u41_():
    return log2(1.0)

@fpy(
    name='Test log2 (2/3)',
    spec=lambda : 1.0,
)
def Test_log2__u40_2_u47_3_u41_():
    return log2(2.0)

@fpy(
    name='Test log2 (3/3)',
    spec=lambda : 2.0,
)
def Test_log2__u40_3_u47_3_u41_():
    return log2(4.0)

@fpy(
    name='Test log1p (1/1)',
    spec=lambda : 0.0,
)
def Test_log1p__u40_1_u47_1_u41_():
    return log1p(0.0)

@fpy(
    name='Test pow (1/5)',
    spec=lambda : 0.0,
)
def Test_pow__u40_1_u47_5_u41_():
    return pow(0.0, 1.0)

@fpy(
    name='Test pow (2/5)',
    spec=lambda : 1.0,
)
def Test_pow__u40_2_u47_5_u41_():
    return pow(1.0, 0.0)

@fpy(
    name='Test pow (3/5)',
    spec=lambda : 1.0,
)
def Test_pow__u40_3_u47_5_u41_():
    return pow(1.0, 1.0)

@fpy(
    name='Test pow (4/5)',
    spec=lambda : 1.0,
)
def Test_pow__u40_4_u47_5_u41_():
    return pow(1.0, 2.0)

@fpy(
    name='Test pow (5/5)',
    spec=lambda : 4.0,
)
def Test_pow__u40_5_u47_5_u41_():
    return pow(2.0, 2.0)

@fpy(
    name='Test sqrt (1/3)',
    spec=lambda : 0.0,
)
def Test_sqrt__u40_1_u47_3_u41_():
    return sqrt(0.0)

@fpy(
    name='Test sqrt (2/3)',
    spec=lambda : 1.0,
)
def Test_sqrt__u40_2_u47_3_u41_():
    return sqrt(1.0)

@fpy(
    name='Test sqrt (3/3)',
    spec=lambda : 2.0,
)
def Test_sqrt__u40_3_u47_3_u41_():
    return sqrt(4.0)

@fpy(
    name='Test cbrt (1/3)',
    spec=lambda : 0.0,
)
def Test_cbrt__u40_1_u47_3_u41_():
    return cbrt(0.0)

@fpy(
    name='Test cbrt (2/3)',
    spec=lambda : 1.0,
)
def Test_cbrt__u40_2_u47_3_u41_():
    return cbrt(1.0)

@fpy(
    name='Test cbrt (3/3)',
    spec=lambda : 2.0,
)
def Test_cbrt__u40_3_u47_3_u41_():
    return cbrt(8.0)

@fpy(
    name='Test hypot (1/2)',
    spec=lambda : 0.0,
)
def Test_hypot__u40_1_u47_2_u41_():
    return hypot(0.0, 0.0)

@fpy(
    name='Test hypot (2/2)',
    spec=lambda : 5.0,
)
def Test_hypot__u40_2_u47_2_u41_():
    return hypot(3.0, 4.0)

@fpy(
    name='Test sin (1/1)',
    spec=lambda : 0.0,
)
def Test_sin__u40_1_u47_1_u41_():
    return sin(0.0)

@fpy(
    name='Test cos (1/1)',
    spec=lambda : 1.0,
)
def Test_cos__u40_1_u47_1_u41_():
    return cos(0.0)

@fpy(
    name='Test tan (1/1)',
    spec=lambda : 0.0,
)
def Test_tan__u40_1_u47_1_u41_():
    return tan(0.0)

@fpy(
    name='Test asin (1/1)',
    spec=lambda : 0.0,
)
def Test_asin__u40_1_u47_1_u41_():
    return asin(0.0)

@fpy(
    name='Test acos (1/1)',
    spec=lambda : 0.0,
)
def Test_acos__u40_1_u47_1_u41_():
    return acos(1.0)

@fpy(
    name='Test atan (1/1)',
    spec=lambda : 0.0,
)
def Test_atan__u40_1_u47_1_u41_():
    return atan(0.0)

@fpy(
    name='Test atan2 (1/1)',
    spec=lambda : 0.0,
)
def Test_atan2__u40_1_u47_1_u41_():
    return atan2(0.0, 1.0)

@fpy(
    name='Test sinh (1/1)',
    spec=lambda : 0.0,
)
def Test_sinh__u40_1_u47_1_u41_():
    return sinh(0.0)

@fpy(
    name='Test cosh (1/1)',
    spec=lambda : 1.0,
)
def Test_cosh__u40_1_u47_1_u41_():
    return cosh(0.0)

@fpy(
    name='Test tanh (1/1)',
    spec=lambda : 0.0,
)
def Test_tanh__u40_1_u47_1_u41_():
    return tanh(0.0)

@fpy(
    name='Test asinh (1/1)',
    spec=lambda : 0.0,
)
def Test_asinh__u40_1_u47_1_u41_():
    return asinh(0.0)

@fpy(
    name='Test acosh (1/1)',
    spec=lambda : 0.0,
)
def Test_acosh__u40_1_u47_1_u41_():
    return acosh(1.0)

@fpy(
    name='Test atanh (1/1)',
    spec=lambda : 0.0,
)
def Test_atanh__u40_1_u47_1_u41_():
    return atanh(0.0)

@fpy(
    name='Test erf (1/1)',
    spec=lambda : 0.0,
)
def Test_erf__u40_1_u47_1_u41_():
    return erf(0.0)

@fpy(
    name='Test erfc (1/1)',
    spec=lambda : 1.0,
)
def Test_erfc__u40_1_u47_1_u41_():
    return erfc(0.0)

@fpy(
    name='Test tgamma (1/3)',
    spec=lambda : 1.0,
)
def Test_tgamma__u40_1_u47_3_u41_():
    return tgamma(1.0)

@fpy(
    name='Test tgamma (2/3)',
    spec=lambda : 1.0,
)
def Test_tgamma__u40_2_u47_3_u41_():
    return tgamma(2.0)

@fpy(
    name='Test tgamma (3/3)',
    spec=lambda : 2.0,
)
def Test_tgamma__u40_3_u47_3_u41_():
    return tgamma(3.0)

@fpy(
    name='Test lgamma (1/2)',
    spec=lambda : 0.0,
)
def Test_lgamma__u40_1_u47_2_u41_():
    return lgamma(1.0)

@fpy(
    name='Test lgamma (2/2)',
    spec=lambda : 0.0,
)
def Test_lgamma__u40_2_u47_2_u41_():
    return lgamma(2.0)

@fpy(
    name='Test ceil (1/4)',
    spec=lambda : 0.0,
)
def Test_ceil__u40_1_u47_4_u41_():
    return ceil(0.0)

@fpy(
    name='Test ceil (2/4)',
    spec=lambda : 1.0,
)
def Test_ceil__u40_2_u47_4_u41_():
    return ceil(0.25)

@fpy(
    name='Test ceil (3/4)',
    spec=lambda : 1.0,
)
def Test_ceil__u40_3_u47_4_u41_():
    return ceil(0.75)

@fpy(
    name='Test ceil (4/4)',
    spec=lambda : 1.0,
)
def Test_ceil__u40_4_u47_4_u41_():
    return ceil(1.0)

@fpy(
    name='Test floor (1/4)',
    spec=lambda : 0.0,
)
def Test_floor__u40_1_u47_4_u41_():
    return floor(0.0)

@fpy(
    name='Test floor (2/4)',
    spec=lambda : 0.0,
)
def Test_floor__u40_2_u47_4_u41_():
    return floor(0.25)

@fpy(
    name='Test floor (3/4)',
    spec=lambda : 0.0,
)
def Test_floor__u40_3_u47_4_u41_():
    return floor(0.75)

@fpy(
    name='Test floor (4/4)',
    spec=lambda : 1.0,
)
def Test_floor__u40_4_u47_4_u41_():
    return floor(1.0)

@fpy(
    name='Test fmod (1/3)',
    spec=lambda : 0.0,
)
def Test_fmod__u40_1_u47_3_u41_():
    return fmod(1.0, 1.0)

@fpy(
    name='Test fmod (2/3)',
    spec=lambda : 0.25,
)
def Test_fmod__u40_2_u47_3_u41_():
    return fmod(1.25, 1.0)

@fpy(
    name='Test fmod (3/3)',
    spec=lambda : 1.0,
)
def Test_fmod__u40_3_u47_3_u41_():
    return fmod(3.0, 2.0)

@fpy(
    name='Test remainder (1/3)',
    spec=lambda : 0.0,
)
def Test_remainder__u40_1_u47_3_u41_():
    return remainder(1, 1)

@fpy(
    name='Test remainder (2/3)',
    spec=lambda : 0.25,
)
def Test_remainder__u40_2_u47_3_u41_():
    return remainder(1.25, 1)

@fpy(
    name='Test remainder (3/3)',
    spec=lambda : -1.0,
)
def Test_remainder__u40_3_u47_3_u41_():
    return remainder(3, 2)

@fpy(
    name='Test fmax (1/5)',
    spec=lambda : 1.0,
)
def Test_fmax__u40_1_u47_5_u41_():
    return fmax(1.0, 0.0)

@fpy(
    name='Test fmax (2/5)',
    spec=lambda : 0.0,
)
def Test_fmax__u40_2_u47_5_u41_():
    return fmax(-1.0, 0.0)

@fpy(
    name='Test fmax (3/5)',
    spec=lambda : 1.0,
)
def Test_fmax__u40_3_u47_5_u41_():
    return fmax(1.0, -1.0)

@fpy(
    name='Test fmax (4/5)',
    spec=lambda : 1.0,
)
def Test_fmax__u40_4_u47_5_u41_():
    return fmax(1.0, NAN)

@fpy(
    name='Test fmax (5/5)',
    spec=lambda : 1.0,
)
def Test_fmax__u40_5_u47_5_u41_():
    return fmax(NAN, 1.0)

@fpy(
    name='Test fmin (1/5)',
    spec=lambda : 0.0,
)
def Test_fmin__u40_1_u47_5_u41_():
    return fmin(1.0, 0.0)

@fpy(
    name='Test fmin (2/5)',
    spec=lambda : -1.0,
)
def Test_fmin__u40_2_u47_5_u41_():
    return fmin(-1.0, 0.0)

@fpy(
    name='Test fmin (3/5)',
    spec=lambda : -1.0,
)
def Test_fmin__u40_3_u47_5_u41_():
    return fmin(1.0, -1.0)

@fpy(
    name='Test fmin (4/5)',
    spec=lambda : 1.0,
)
def Test_fmin__u40_4_u47_5_u41_():
    return fmin(1.0, NAN)

@fpy(
    name='Test fmin (5/5)',
    spec=lambda : 1.0,
)
def Test_fmin__u40_5_u47_5_u41_():
    return fmin(NAN, 1.0)

@fpy(
    name='Test fdim (1/3)',
    spec=lambda : 1.0,
)
def Test_fdim__u40_1_u47_3_u41_():
    return fdim(2.0, 1.0)

@fpy(
    name='Test fdim (2/3)',
    spec=lambda : 0.0,
)
def Test_fdim__u40_2_u47_3_u41_():
    return fdim(1.0, 1.0)

@fpy(
    name='Test fdim (3/3)',
    spec=lambda : 0.0,
)
def Test_fdim__u40_3_u47_3_u41_():
    return fdim(1.0, 2.0)

@fpy(
    name='Test copysign (1/4)',
    spec=lambda : 1.0,
)
def Test_copysign__u40_1_u47_4_u41_():
    return copysign(1.0, 1.0)

@fpy(
    name='Test copysign (2/4)',
    spec=lambda : 1.0,
)
def Test_copysign__u40_2_u47_4_u41_():
    return copysign(-1.0, 1.0)

@fpy(
    name='Test copysign (3/4)',
    spec=lambda : -1.0,
)
def Test_copysign__u40_3_u47_4_u41_():
    return copysign(1.0, -1.0)

@fpy(
    name='Test copysign (4/4)',
    spec=lambda : -1.0,
)
def Test_copysign__u40_4_u47_4_u41_():
    return copysign(-1.0, -1.0)

@fpy(
    name='Test trunc (1/4)',
    spec=lambda : 0.0,
)
def Test_trunc__u40_1_u47_4_u41_():
    return trunc(0.0)

@fpy(
    name='Test trunc (2/4)',
    spec=lambda : 0.0,
)
def Test_trunc__u40_2_u47_4_u41_():
    return trunc(0.25)

@fpy(
    name='Test trunc (3/4)',
    spec=lambda : 0.0,
)
def Test_trunc__u40_3_u47_4_u41_():
    return trunc(-0.75)

@fpy(
    name='Test trunc (4/4)',
    spec=lambda : 1.0,
)
def Test_trunc__u40_4_u47_4_u41_():
    return trunc(1.0)

@fpy(
    name='Test round (1/4)',
    spec=lambda : 0.0,
)
def Test_round__u40_1_u47_4_u41_():
    return round(0.0)

@fpy(
    name='Test round (2/4)',
    spec=lambda : 0.0,
)
def Test_round__u40_2_u47_4_u41_():
    return round(0.25)

@fpy(
    name='Test round (3/4)',
    spec=lambda : 1.0,
)
def Test_round__u40_3_u47_4_u41_():
    return round(0.75)

@fpy(
    name='Test round (4/4)',
    spec=lambda : 1.0,
)
def Test_round__u40_4_u47_4_u41_():
    return round(1.0)

@fpy(
    name='Test nearbyint (1/4)',
    spec=lambda : 0.0,
)
def Test_nearbyint__u40_1_u47_4_u41_():
    return nearbyint(0.0)

@fpy(
    name='Test nearbyint (2/4)',
    spec=lambda : 0.0,
)
def Test_nearbyint__u40_2_u47_4_u41_():
    return nearbyint(0.25)

@fpy(
    name='Test nearbyint (3/4)',
    spec=lambda : 1.0,
)
def Test_nearbyint__u40_3_u47_4_u41_():
    return nearbyint(0.75)

@fpy(
    name='Test nearbyint (4/4)',
    spec=lambda : 1.0,
)
def Test_nearbyint__u40_4_u47_4_u41_():
    return nearbyint(1.0)

