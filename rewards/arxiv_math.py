"""This module provides functions for parsing mathematical expressions from text."""

import re
from enum import Enum
from fractions import Fraction
from functools import total_ordering
from typing import Any, Optional
from sympy import simplify, Symbol, Mul

import regex
import sympy
from loguru import logger
from sympy import N
from sympy.parsing.latex import parse_latex


manual_mapper = {
    "00009 \\sqrt{15}": "9\\sqrt{15}",
    "The\\ area\\ of\\ triangle\\ ABC\\ is\\ \\frac{54\\sqrt{6}}{25}": "\\frac{54\\sqrt{6}}{25}",
    "-\\dfrac{\\pi\\,e}{2}": "-\\dfrac{\\pi e}{2}",
    "\n\\begin{matrix}\n- \\\\ 1013 \\\\ \\\\ \\times \\\\ \\\\ \\ln \\\\ \\\\ \\left( \\\\ 2026 \\\\ \\right) \\end{matrix}\n": "-1013 \\times \\ln(2026)",
    "\n\\begin{matrix}\n- \\\\ 1013 \\\\ \\\\ \\times \\\\ \\\\ \\ln \\\\ \\\\ 2026 \\end{matrix}\n": "-1013 \\times \\ln(2026)",
    "\n\\begin{matrix}\n\\frac{ \\\\ 4 \\\\ \\pi \\\\ }{ \\\\ 3 \\\\ } \\\\ + \\\\ \\\\ \\sqrt{ \\\\ \\, \\\\ 3 \\\\ \\, \\\\ } \\\\ \\end{matrix}\n": "\\frac{4\\pi}{3} + \\sqrt{3}",
    "\n\\begin{matrix}\n\\frac{ \\\\ 4 \\\\ \\pi \\\\ }{ \\\\ 3 \\\\ } \\\\ + \\\\ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ \\end{matrix}\n": "\\frac{4\\pi}{3} + \\sqrt{3}",
    "\n\\begin{matrix}\n\\frac{ \\\\ s \\\\ }{ \\\\ r \\\\ } \\\\ = \\\\ \\\\ \\frac{ \\\\ 2 \\\\ \\sqrt{ \\\\ 6 \\\\ } \\\\ \\\\ }{ \\\\ 3 \\\\ } \\end{matrix}\n": "\\frac{2\\sqrt{6}}{3}",
    "\n\\begin{matrix}\n\\frac{ \\\\ 54 \\\\ \\sqrt{ \\\\ \\, \\\\ 6 \\\\ \\, \\\\ } \\\\ }{ \\\\ 25 \\\\ } \\end{matrix}\n": "\\frac{54\\sqrt{6}}{25}",
    "\\begin{pmatrix} a \\\\ 13 \\\\ b \\\\ 14 \\\\ c \\\\ 18 \\end{pmatrix}": "13,14,18",
    "\\begin{pmatrix} a \\\\ 13 \\\\ b \\\\ 14 \\\\ c \\\\ 18 \\\\ \\end{pmatrix}": "13,14,18",
    "\n\\begin{matrix}\n\\frac{ \\\\ 5 \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\\\ + \\\\ \\\\ \\frac{ \\\\ \\sqrt{ \\\\ 6 \\\\ } \\\\ + \\\\ \\\\ \\sqrt{ \\\\ 2 \\\\ } \\\\ \\\\ }{ \\\\ 4 \\\\ } \\end{matrix}\n": "\\frac{5\\sqrt{3}}{2} + \\frac{\\sqrt{6} + \\sqrt{2}}{4}",
    "\n\\begin{matrix}\n\\frac{ \\\\ 10 \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ + \\\\ \\sqrt{ \\\\ 6 \\\\ } \\\\ + \\\\ \\sqrt{ \\\\ 2 \\\\ } \\\\ \\, \\\\ }{ \\\\ 4 \\\\ } \\end{matrix}\n": "\\frac{10\\sqrt{3} + \\sqrt{6} + \\sqrt{2}}{4}",
    "\n\\begin{matrix}\n\\text{the} \\\\ \\\\ \\text{radius} \\\\ \\\\ \\text{of} \\\\ \\\\ \\omega \\\\ _{1} \\\\ \\\\ \\text{is} \\\\ \\\\ \\\\ \\dfrac{ \\\\ 288 \\\\ \\, \\\\ \\sqrt{ \\\\ \\, \\\\ 527 \\\\ \\, \\\\ } \\\\ \\, \\\\ } \\\\ \\\\ \\dfrac{ \\\\ \\, \\\\ } \\\\ \\\\ 527 \\\\ \\\\ \\end{matrix}\n": "\\frac{288\\sqrt{527}}{527}",
    "\n\\begin{matrix}\n\\text{the} \\\\ \\\\ \\text{radius} \\\\ \\\\ \\text{of} \\\\ \\\\ \\omega_{1} \\\\ \\\\ = \\\\ \\\\ \\\\ \\dfrac{ \\\\ 288 \\\\ \\sqrt{ \\\\ 527 \\\\ } \\\\ \\\\ }{ \\\\ 527 \\\\ } \\end{matrix}\n": "\\frac{288\\sqrt{527}}{527}",
    "\n\\begin{matrix}\n\\frac{ \\\\ \\sin \\\\ \\left( \\\\ \\\\ \\dfrac{ \\\\ \\, \\\\ 6 \\\\ \\, \\\\ \\, \\\\ \\pi \\\\ \\, \\\\ \\, \\\\ } \\\\ \\\\ 13 \\\\ \\\\ \\right) \\\\ \\\\ } \\\\ \\\\ / \\\\ \\\\ \\\\ \\sin \\\\ \\\\ \\left( \\\\ \\\\ \\dfrac{ \\\\ \\, \\\\ \\pi \\\\ \\, \\\\ \\, \\\\ } \\\\ \\\\ 13 \\\\ \\\\ \\right) \\\\ \\\\ \\end{matrix}\n": "\\frac{\\sin(\\frac{6\\pi}{13})}{\\sin(\\frac{\\pi}{13})}",
    "\n\\begin{matrix} 3 \\\\ 3 \\\\ 7 \\\\ 5 \\end{matrix}\n": "3375",
    "\n\\begin{matrix}\n\\frac{ \\\\ 1311 \\\\ }{ \\\\ 2017 \\\\ } \\end{matrix}\n": "\\frac{1311}{2017}",
    "\n\\begin{matrix}\n\\text{ } \\\\ \\dfrac{ \\\\ 9 \\\\ \\, \\\\ \\sqrt{ \\\\ \\, \\\\ 23 \\\\ \\, \\\\ } \\\\ \\, \\\\ } \\\\ \\dfrac{ \\\\ \\, \\\\ } \\\\ { \\\\ 23 \\\\ } \\end{matrix}\n": "\\frac{9\\sqrt{23}}{23}",
    "\n\\begin{matrix}\n\\frac{ \\\\ 9 \\\\ \\sqrt{ \\\\ 23 \\\\ } \\\\ }{ \\\\ 23 \\\\ } \\end{matrix}\n": "\\frac{9\\sqrt{23}}{23}",
    "\n\\begin{matrix}\n1 \\\\ - \\\\ \\dfrac{ \\\\ 2 \\\\ }{ \\\\ \\pi \\\\ } \\end{matrix}\n": "1 - \\frac{2}{\\pi}",
    "\n\\begin{matrix}\n1 \\\\ - \\\\ \\dfrac{ \\\\ 2 \\\\ \\\\ }{ \\\\ \\pi \\\\ } \\end{matrix}\n": "1 - \\frac{2}{\\pi}",
    "\n\\begin{matrix}\n\\dfrac{ \\\\ - \\, 1 \\\\ + \\\\ \\sqrt{ \\\\  \\, 17 \\\\ } \\\\ \\, \\\\ }{ \\\\ 2 \\\\ } \\\\ \\\\ , \\\\ \\\\ \\\\ \\dfrac{ \\\\ - \\, 1 \\\\ - \\\\ \\sqrt{ \\\\  \\, 17 \\\\ } \\\\ \\, \\\\ }{ \\\\ 2 \\\\ } \\end{matrix}\n": "\\frac{-1 + \\sqrt{17}}{2}, \\frac{-1 - \\sqrt{17}}{2}",
    "\n\\begin{matrix}\n8 \\\\ \\\\ \\times \\\\ \\\\ \\sqrt{ \\\\ 10 \\\\ } \\end{matrix}\n": "8\\sqrt{10}",
    "\n\\begin{matrix}\n\\text{ } \\\\ \\sqrt{ \\\\ 23 \\\\ } \\\\ - \\\\ \\\\ 2 \\\\ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ \\text{ } \\end{matrix}\n": "\\sqrt{23} - 2\\sqrt{3}",
    "\\,R\\text{ is the unique positive root of }2R^{3}-17R-15=0\n\\;(\\;R\\approx 3.284\\;)": "2\\sqrt{\\frac{17}{6}}\\cos(\\frac{1}{3}\\arccos(\\tfrac{45\\sqrt{6}}{34\\sqrt{17}}))",  # model mentions this right before.
    "R\\approx3.283532\\,.": "(1/3)\\arccos((45\\sqrt{6})/(34\\sqrt{17}))",
    "\\frac2\\pi-1": "\\frac{2}{\\pi}-1",
    "0002^{99}": "2^{99}",
    "\n\\begin{matrix}\n\\\\ 2 \\\\ \\\\ ^{} \\\\ \\\\ 99 \\\\ \\\\ \\end{matrix}\n": "2^{99}",
    "\n\\begin{matrix}\n \\\\ 3 \\\\ + \\\\  \\\\ 2 \\\\ \\\\ \\sqrt{ \\\\  \\\\ 2 \\\\ } \\\\ \\end{matrix}\n": "3 + 2\\sqrt{2}",
    "\\begin{pmatrix} \\\\ 2 \\\\ \\\\ 3 \\\\ \\\\ \\dfrac{ \\\\ 3 \\\\ }{ \\\\ 2 \\\\ } \\\\ + \\\\ \\\\ i \\\\ \\\\ \\dfrac{ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\\\ \\\\ \\dfrac{ \\\\ 3 \\\\ }{ \\\\ 2 \\\\ } \\\\ - \\\\ \\\\ i \\\\ \\\\ \\dfrac{ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\end{pmatrix}": "2,3,\\frac{3}{2} + i\\frac{\\sqrt{3}}{2}, \\frac{3}{2} - i\\frac{\\sqrt{3}}{2}",
    "\\begin{pmatrix} \\\\ 2 \\\\ \\\\ 3 \\\\ \\\\ \\dfrac{ \\\\ 3 \\\\ }{ \\\\ 2 \\\\ } \\\\ + \\\\ \\\\ i \\\\ \\\\ \\dfrac{ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\\\ \\\\ \\\\ \\dfrac{ \\\\ 3 \\\\ }{ \\\\ 2 \\\\ } \\\\ - \\\\ \\\\ i \\\\ \\\\ \\dfrac{ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\end{pmatrix}": "2,3,\\frac{3}{2} + i\\frac{\\sqrt{3}}{2}, \\frac{3}{2} - i\\frac{\\sqrt{3}}{2}",
    "\\begin{pmatrix} \\\\ 2 \\\\ \\\\ 3 \\\\ \\\\ \\\\ 2 \\\\ + \\\\ \\\\ \\omega \\\\ \\\\ \\\\ \\\\ 2 \\\\ + \\\\ \\\\ \\\\ \\omega^{2} \\\\ \\\\ \\end{pmatrix}": "2,3,\\frac{3}{2} + i\\frac{\\sqrt{3}}{2}, \\frac{3}{2} - i\\frac{\\sqrt{3}}{2}",
    "\\begin{pmatrix} \\\\ 2 \\\\ \\\\ 3 \\\\ \\\\ \\dfrac{ \\\\ 3 \\\\ }{ \\\\ 2 \\\\ } \\\\ + \\\\ \\\\ i \\\\ \\\\ \\dfrac{ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\\\ \\\\ \\\\ \\dfrac{ \\\\ 3 \\\\ }{ \\\\ 2 \\\\ } \\\\ - \\\\ \\\\ i \\\\ \\\\ \\dfrac{ \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ }{ \\\\ 2 \\\\ } \\\\ \\end{pmatrix}": "2,3,\\frac{3}{2} + i\\frac{\\sqrt{3}}{2}, \\frac{3}{2} - i\\frac{\\sqrt{3}}{2}",
    "\n\\begin{matrix}\nx \\\\ = \\\\ \\\\ \\sqrt{ \\\\ 4 \\\\ + \\\\ \\\\ 2 \\\\ \\sqrt{ \\\\ 3 \\\\ } \\\\ } \\end{matrix}\n": "\\sqrt{4 + 2\\sqrt{3}}",
    "a = 2, a = 3, a = \\frac{3 + i\\sqrt{3}}{2}, a = \\frac{3 - i\\sqrt{3}}{2}": "2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}",
    "a = 2, a = 3, a = \\frac{3 + i\\sqrt{3}}{2}, \\text{ or } a = \\frac{3 - i\\sqrt{3}}{2}": "2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}",
    "2\n\n3\n\n3/2+(\u221a3/2)i\n\n3/2-(\u221a3/2)i": "2,3,3/2+(\u221a3/2)i,3/2-(\u221a3/2)i",
    "\n\\begin{matrix}\n\\text{ } \\\\ F_{ \\\\ 30 \\\\ } \\\\ - \\\\ 2 \\\\ \\\\ \\times \\\\ \\\\ \\sqrt{ \\\\ \\\\ \\log \\\\ _ \\\\ 3 \\\\ \\\\ 2 \\\\ } \\end{matrix}\n": "(832040 - 2)\\sqrt{\\log_{3}2}",
    "\n\\begin{matrix}\n\\text{ } \\\\ \\sqrt{ \\\\ \\log \\\\ _{ \\\\ 2 \\\\ } \\\\ P \\\\ } \\\\ = \\\\ \\\\ ( \\\\ F \\\\ _{ \\\\ 30 \\\\ } \\\\ \\\\ - \\\\ \\\\ 2 \\\\ ) \\\\ \\\\ \\sqrt{ \\\\ \\\\ \\log \\\\ _{ \\\\ 3 \\\\ } \\\\ \\\\ 2 \\\\ } \\end{matrix}\n": "(832040 - 2)\\sqrt{\\log_{3}2}",
    "\n\\begin{matrix}\n\\frac{ \\\\ F_{ \\\\ 30 \\\\ } \\\\ - \\\\ 2 \\\\ \\, \\\\ } \\\\ \\\\ / \\\\ \\\\ \\sqrt{ \\\\ \\, \\\\ \\log_{ \\\\ 2 \\\\ } \\\\ \\, \\\\ 3 \\\\ \\, \\\\ } \\\\ \\end{matrix}\n": "\\frac{832040 - 2}{\\sqrt{\\log_{2}3}}",
    "intersect at $y=k=506.25$": "intersect at $y=k=\\boxed{506.25}$",
    "\\dfrac{1}{2\\sin\\!\\bigl(\\pi/26\\bigr)} \\quad\\text{(approximately }4.14812\\text{)}": "\\frac{1}{2\\sin(\\frac{\\pi}{26})}}",
    "\\{\\,2,\\ 3,\\ \\tfrac{3}{2}\\pm\\tfrac{\\sqrt{3}}{2}i\\,\\}": "\\{2,3,\\frac{3}{2}+\\frac{\\sqrt{3}*i}{2},\\frac{3}{2}-\\frac{\\sqrt{3}*i}{2}\\}",
    "\\{\\,2,\\ 3,\\ 2+e^{2\\pi i/3},\\ 2+e^{4\\pi i/3}\\,\\}=\\{\\,2,\\ 3,\\ \\tfrac{3}{2}\\pm \\tfrac{\\sqrt{3}}{2}i\\,\\}": "\\{2,3,\\frac{3}{2}+\\frac{\\sqrt{3}*i}{2},\\frac{3}{2}-\\frac{\\sqrt{3}*i}{2}\\}",
    "\\displaystyle \\frac{737}{39}\\;\\text{(about }18.9\\text{ cards)}": "\\frac{737}{39}",
    "\\frac{\\binom{2024}{990}+23}{46}": "\\frac{1}{46}\\binom{2024}{990}+\\frac{1}{2}",
    "2, 3, \\frac{3 \pm i\\sqrt{3}}{2}": "2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}",
    "2, 3, \\frac{3 \\pm i \\sqrt{3}}{2}": "2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}",
    "(N+1)^2(4N+1)": "4N^3+9N^2+6N+1",
    "(N+1)^2 (4N+1)": "4N^3+9N^2+6N+1",
    "a\\in\\left\\{2,\\ 3,\\ 2+\\omega,\\ 2+\\omega^2\\right\\}": "2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}",
    "\\frac{n(n-1)}2": "\\frac{n^2 - n}{2}",
    "\\text{Bob wins exactly for those }G\\text{ with }\\forall p,\\ \\bigoplus_i a_{p,i}=0,\\ \\text{ and the requested sum is }505.": "505"
}

# maps a full response to the answer, in case no \boxed{} is present
complete_mapper = {
    "\n<|begin_of_box|>A<|end_of_box|>": "\\boxed{A}",
    "\n<|begin_of_box|>B<|end_of_box|>": "\\boxed{B}",
    "\n<|begin_of_box|>C<|end_of_box|>": "\\boxed{C}",
    "\n<|begin_of_box|>D<|end_of_box|>": "\\boxed{D}",
    "\n<|begin_of_box|>E<|end_of_box|>": "\\boxed{E}",
    """Let\u2019s write the denominator as\nden(x)=x^3\u22126x^2+11x\u22126=(x\u22121)(x\u22122)(x\u22123).  Then near x=1 we set\n\ng(x)= (2025+x)\u2009ln(2025+x)   /  [(x\u22122)(x\u22123)],\n\nso that\n\nf(x)= g(x)/(x\u22121).\n\nNow g(x) is analytic at x=1, and g(1)= (2026\u2009ln\u20092026)/[(1\u22122)(1\u22123)]\n= (2026\u2009ln\u20092026)/( (\u22121)(\u22122) )\n= (2026\u2009ln\u20092026)/2.\n\nHence near x=0 the simple pole at x=1 contributes the expansion\n\ng(1)/(x\u22121)= \u2212g(1)\u00b71/(1\u2212x)= \u2212g(1)\u00b7\u2211_{n\u22650}x^n.\n\nThus for large n the Taylor\u2010coefficient a_n=d_n tends to \u2212g(1).  Therefore\n\nlimit\u2009d_n = \u2212g(1)\n= \u2212 (2026\u2009ln\u20092026)/2\n= \u22121013\u2009ln\u20092026.\n\n\u25a1""": "\\boxed{-1013\\ln(2026)}",
    "Let  \n\nB=(0,0),\u2003C=(3+x,0),\u2003D=(3,0),\u2003A=(3,h)\u2003(h>0),  \n\nso BD=3,\u2003DC=x,\u2003AD=h.  \nLet the circum-center be O=(u,v).  \n\n1.  OA\u00b2=OB\u00b2 gives  \n\n   (u\u22123)\u00b2+(v\u2212h)\u00b2=u\u00b2+v\u00b2 \u27f9 6u+2hv = 9+h\u00b2 .\u2003(1)\n\n2.  OA\u00b2=OC\u00b2 gives  \n\n   (u\u22123)\u00b2+(v\u2212h)\u00b2 =(u\u22123\u2212x)\u00b2+v\u00b2 \u27f9 (x+3)(h\u00b2\u22122hv)=3x .\u2003(2)\n\n   (2) can be rewritten as      h\u00b2\u22122hv =3x.  \n   Denote S:=h\u00b2\u22122hv, so      S=3x \u2003(x=S/3).                                (3)\n\n3.  OD=2 gives  \n\n   (u\u22123)\u00b2+v\u00b2 =4 .                                                        (4)\n\n4.  Area(ABD)=Area(AOC).  \n   Area(ABD)=\u00bd\u00b7BD\u00b7AD = \u00bd\u00b73h.  \n\n   For AOC,  \n      |(A\u2212O)\u00d7(C\u2212O)| = |\u2212v(3\u2212u) \u2212 (h\u2212v)(3+x\u2212u)|  \n                     = |v(2u\u22123)\u2212hu| .\n\n   Hence |v(2u\u22123)\u2212hu| = 3h.\u2003(5)\n\n--------------------------------------------------------------------\nExpress u and v with h and S\n\nFrom (1):\u2003u = (9+S)/6.                                             (6)\n\nUsing S:=h\u00b2\u22122hv \u21d2 v = (h\u00b2\u2212S)/(2h).                                (7)\n\nNotice that  \n      2u\u22123 = 2(9+S)/6 \u22123 = S/3,\u2003    3+x\u2212u = u.               \n\n--------------------------------------------------------------------\nArea equality\n\nWith (6)\u2013(7), equation (5) becomes  \n\n|  (S/3)\u00b7v \u2212 h\u00b7u | = 3h  \n\u27f9 |  (S/3)\u00b7v \u2212 h\u00b7(9+S)/6 | = 3h  \n\u27f9 |2Sv \u2212 h(9+S)| = 18h.                                          (8)\n\nSubstituting v from (7) into (8) and simplifying gives  \n\n(S\u00b2+9h\u00b2)\u00b2 = 324h\u2074 \u27f9 S\u00b2 = 9h\u00b2 \u27f9 S = \u00b13h .                       (9)\n\n--------------------------------------------------------------------\nReject S=\u22123h\n\nIf S=\u22123h, then from (3) x = \u2212h <0, impossible.  \nHence S = 3h.                                                    (10)\n\n--------------------------------------------------------------------\nDetermine h from OD\n\nWith S=3h, equations (6)\u2013(7) give  \n\nu = (9+3h)/6 ,\u2003\u2003v = (h\u22123)/2.                                      \n\nInsert these in (4):\n\n[(9+3h)/6 \u22123]\u00b2 + [(h\u22123)/2]\u00b2 = 4  \n\u27f9 [(3h\u22129)/6]\u00b2 + [(h\u22123)/2]\u00b2 = 4  \n\u27f9 18(h\u22123)\u00b2 = 144  \n\u27f9 (h\u22123)\u00b2 = 8  \n\u27f9 h\u22123 = 2\u221a2 \u2003(since h>3).  \n\nThus\u2003h = 3 + 2\u221a2.                                               (11)\n\n--------------------------------------------------------------------\nAltitude AD\n\nAD = h = 3 + 2\u221a2.": "\\boxed{3 + 2\\sqrt{2}}",
    """We are given: Define operation \u2295 by:\nx \u2295 y = x y - 2x - 2y + 6.\nCompute all complex numbers a such that:\na \u2295 (a \u2295 (a \u2295 a)) = a.\n\nLet's denote: f(x,y) = x \u2295 y = xy - 2x - 2y + 6.\nWe want: a \u2295 (a \u2295 (a \u2295 a)) = a.\nLet's compute step by step.\n\nLet b = a \u2295 a.\nThen b = a*a - 2a - 2a + 6 = a^2 - 4a + 6.\nThen c = a \u2295 b = a \u2295 (a \u2295 a) = a * b - 2a - 2b + 6 = a(a^2 - 4a + 6) - 2a - 2(a^2 - 4a + 6) + 6 = a^3 - 4a^2 + 6a - 2a - 2a^2 + 8a - 12 + 6 = a^3 - 6a^2 + (6a-2a+8a) + (-12+6) = a^3 - 6a^2 + 12a - 6.\nThen d = a \u2295 c = a \u2295 (a \u2295 (a \u2295 a)) = a * c - 2a - 2c + 6 = a(a^3 - 6a^2 + 12a - 6) - 2a - 2(a^3 - 6a^2 + 12a - 6) + 6 = a^4 - 6a^3 + 12a^2 - 6a - 2a - 2a^3 + 12a^2 - 24a + 12 + 6 = a^4 - 8a^3 + (12a^2+12a^2) + (-6a-2a-24a) + (12+6) = a^4 - 8a^3 + 24a^2 - 32a + 18.\nSo equation: a^4 - 8a^3 + 24a^2 - 32a + 18 = a.\nSo a^4 - 8a^3 + 24a^2 - 33a + 18 = 0.\nSo polynomial: a^4 - 8a^3 + 24a^2 - 33a + 18 = 0.\nWe want to factor it. Possibly it factors as (a^2 + ...)? Alternatively, note pattern: (a-2)^4 = a^4 - 8a^3 + 24a^2 - 32a + 16. So our polynomial is (a-2)^4 - 17a + 2? Actually: (a-2)^4 = a^4 - 8a^3 + 24a^2 - 32a + 16. Then subtract gives: (a-2)^4 - (a^4 - 8a^3+24a^2-33a+18) = (a^4-8a^3+24a^2-32a+16) - (a^4-8a^3+24a^2-33a+18) = a - 2. So then a^4-8a^3+24a^2-33a+18 = (a-2)^4 - (a-2) = (a-2)[(a-2)^3 - 1] = (a-2)[(a-2)^3 - 1]. And (a-2)^3 - 1 factors as (a-2 - 1)[(a-2)^2 + (a-2) + 1] = (a-3)[(a-2)^2 + (a-2) + 1]. So then polynomial becomes: (a-2)(a-3)(a^2 - 4a + 4 + a - 2 + 1) = (a-2)(a-3)(a^2 - 3a + 3). So roots: a = 2, a = 3, and a = [3 \u00b1 sqrt(9 - 12)]/2 = (3 \u00b1 i\u221a3)/2.\nBut wait: Check: a=2: then a \u2295 a = 4 - 4 - 4 + 6 = 2. So a \u2295 a = 2. Then a \u2295 (a \u2295 a) = 2 \u2295 2 = 4 - 4 - 4 + 6 = 2. Then a \u2295 (a \u2295 (a \u2295 a)) = 2 \u2295 2 = 2. So works.\na=3: then a \u2295 a = 9 - 6 - 6 + 6 = 3. So then a \u2295 (a \u2295 a)=3 \u2295 3 = 9 - 6 - 6 + 6 = 3. So works.\nFor a = (3 \u00b1 i\u221a3)/2, then check? Possibly they also work. But careful: The operation is defined on complex numbers. So all complex numbers are allowed. So answer: a = 2, 3, (3 \u00b1 i\u221a3)/2.\nBut wait: There might be additional conditions? Problem says \"Compute all complex numbers a such that ...\" So answer: \\boxed{2}, \\boxed{3}, \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}.\n\nBut check: (3 + i\u221a3)/2. Then compute a \u2295 a? Possibly it gives something. But it's fine.\n\nThus final answer: \\boxed{2}, \\boxed{3}, \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}.</think>### Step 1. Define the Operation\n\nThe operation \u2295 is defined as:\n\n\\[\nx \\oplus y = xy - 2x - 2y + 6\n\\]\n\nWe are to find all complex numbers \\( a \\) such that:\n\n\\[\na \\oplus (a \\oplus (a \\oplus a)) = a\n\\]\n\n---\n\n### Step 2. Compute the Inner Expressions\n\nLet\u2019s compute step by step:\n\n- **First, compute \\( a \\oplus a \\):**\n\n\\[\na \\oplus a = a \\cdot a - 2a - 2a + 6 = a^2 - 4a + 6\n\\]\n\n- **Next, compute \\( a \\oplus (a \\oplus a) \\):**\n\nLet \\( b = a^2 - 4a + 6 \\). Then:\n\n\\[\na \\oplus b = ab - 2a - 2b + 6\n\\]\n\nSubstitute \\( b \\):\n\n\\[\na \\oplus (a \\oplus a) = a(a^2 - 4a + 6) - 2a - 2(a^2 - 4a + 6) + 6\n\\]\n\n\\[\n= a^3 - 4a^2 + 6a - 2a - 2a^2 + 8a - 12 + 6\n\\]\n\n\\[\n= a^3 - 6a^2 + 12a - 6\n\\]\n\n- **Now compute \\( a \\oplus (a \\oplus (a \\oplus a)) \\):**\n\nLet \\( c = a^3 - 6a^2 + 12a - 6 \\). Then:\n\n\\[\na \\oplus c = ac - 2a - 2c + 6\n\\]\n\nSubstitute \\( c \\):\n\n\\[\na(a^3 - 6a^2 + 12a - 6) - 2a - 2(a^3 - 6a^2 + 12a - 6) + 6\n\\]\n\n\\[\n= a^4 - 6a^3 + 12a^2 - 6a - 2a - 2a^3 + 12a^2 - 24a + 12 + 6\n\\]\n\n\\[\n= a^4 - 8a^3 + 24a^2 - 32a + 18\n\\]\n\n---\n\n### Step 3. Set Up the Equation\n\nWe require:\n\n\\[\na \\oplus (a \\oplus (a \\oplus a)) = a\n\\]\n\nSo:\n\n\\[\na^4 - 8a^3 + 24a^2 - 32a + 18 = a\n\\]\n\nSubtract \\( a \\) from both sides:\n\n\\[\na^4 - 8a^3 + 24a^2 - 33a + 18 = 0 \\tag{1}\n\\]\n\n---\n\n### Step 4. Factor the Polynomial\n\nNote that:\n\n\\[\n(a - 2)^4 = a^4 - 8a^3 + 24a^2 - 32a + 16\n\\]\n\nCompare with equation (1):\n\n\\[\na^4 - 8a^3 + 24a^2 - 33a + 18 = (a - 2)^4 - (a - 2)\n\\]\n\n\\[\n= (a - 2)\\left[(a - 2)^3 - 1\\right]\n\\]\n\nFactor further:\n\n\\[\n(a - 2)^3 - 1 = (a - 3)\\left[(a - 2)^2 + (a - 2) + 1\\right]\n\\]\n\nSo the full factorization is:\n\n\\[\n(a - 2)(a - 3)(a^2 - 3a + 3)\n\\]\n\n---\n\n### Step 5. Find the Roots\n\n- From \\( a - 2 = 0 \\):\u2003\\( a = 2 \\)\n- From \\( a - 3 = 0 \\):\u2003\\( a = 3 \\)\n- From \\( a^2 - 3a + 3 = 0 \\):\n\n\\[\na = \\frac{3 \\pm \\sqrt{9 - 12}}{2} = \\frac{3 \\pm i\\sqrt{3}}{2}\n\\]\n\n---\n\n### \u2705 Final Answer\n\n\\[\n\\boxed{2},\\quad \\boxed{3},\\quad \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}\n\\]""": "\\boxed{2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}}",
    """### Step 1. Understand the Operation\n\nThe operation is defined as:\n\\[\nx \\oplus y = xy - 2x - 2y + 6\n\\]\nThis can be rewritten as:\n\\[\nx \\oplus y = (x - 2)(y - 2) + 2\n\\]\n\n---\n\n### Step 2. Compute the Nested Expression\n\nWe want to compute:\n\\[\na \\oplus (a \\oplus (a \\oplus a))\n\\]\n\nLet\u2019s compute step by step:\n\n- **First level**:  \n  \\[\n  a \\oplus a = (a - 2)^2 + 2\n  \\]\n\n- **Second level**:  \n  \\[\n  a \\oplus (a \\oplus a) = (a - 2)^3 + 2\n  \\]\n\n- **Third level**:  \n  \\[\n  a \\oplus (a \\oplus (a \\oplus a)) = (a - 2)^4 + 2\n  \\]\n\nSo the equation becomes:\n\\[\n(a - 2)^4 + 2 = a\n\\]\n\n---\n\n### Step 3. Simplify the Equation\n\nSubtract 2 from both sides:\n\\[\n(a - 2)^4 = a - 2\n\\]\n\nLet \\( u = a - 2 \\). Then the equation becomes:\n\\[\nu^4 = u \\quad \\Rightarrow \\quad u^4 - u = 0\n\\]\n\nFactor:\n\\[\nu(u^3 - 1) = 0\n\\]\n\nSo either:\n- \\( u = 0 \\), or  \n- \\( u^3 = 1 \\)\n\n---\n\n### Step 4. Solve for \\( u \\)\n\n- If \\( u = 0 \\), then \\( a = 2 \\)\n- If \\( u^3 = 1 \\), then the cube roots of unity are:\n  \\[\n  u = 1, \\quad u = \\omega = -\\frac{1}{2} + \\frac{i\\sqrt{3}}{2}, \\quad u = \\omega^2 = -\\frac{1}{2} - \\frac{i\\sqrt{3}}{2}\n  \\]\n  Then:\n  \\[\n  a = u + 2 \\Rightarrow a = 3, \\quad a = 2 + \\omega, \\quad a = 2 + \\omega^2\n  \\]\n\nCompute:\n\\[\n2 + \\omega = \\frac{3}{2} + \\frac{i\\sqrt{3}}{2}, \\quad 2 + \\omega^2 = \\frac{3}{2} - \\frac{i\\sqrt{3}}{2}\n\\]\n\n---\n\n### Step 5. Final Answer\n\nThe complex numbers \\( a \\) satisfying the equation are:\n\\[\n\\boxed{2}, \\quad \\boxed{3}, \\quad \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}\n\\]""": "\\boxed{2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}}",
    """### Step 1. Compute \\( a \\oplus a \\)\n\nThe operation is defined as:\n\\[\nx \\oplus y = xy - 2x - 2y + 6\n\\]\n\nSo,\n\\[\na \\oplus a = a \\cdot a - 2a - 2a + 6 = a^2 - 4a + 6\n\\]\n\n---\n\n### Step 2. Compute \\( a \\oplus (a \\oplus a) \\)\n\nLet \\( b = a \\oplus a = a^2 - 4a + 6 \\). Then:\n\\[\na \\oplus b = a \\cdot b - 2a - 2b + 6\n\\]\nSubstitute \\( b \\):\n\\[\na \\oplus (a \\oplus a) = a(a^2 - 4a + 6) - 2a - 2(a^2 - 4a + 6) + 6\n\\]\nSimplify:\n\\[\n= a^3 - 4a^2 + 6a - 2a - 2a^2 + 8a - 12 + 6\n\\]\n\\[\n= a^3 - 6a^2 + 12a - 6\n\\]\n\n---\n\n### Step 3. Compute \\( a \\oplus (a \\oplus (a \\oplus a)) \\)\n\nLet \\( c = a^3 - 6a^2 + 12a - 6 \\). Then:\n\\[\na \\oplus c = a \\cdot c - 2a - 2c + 6\n\\]\nSubstitute \\( c \\):\n\\[\n= a(a^3 - 6a^2 + 12a - 6) - 2a - 2(a^3 - 6a^2 + 12a - 6) + 6\n\\]\n\\[\n= a^4 - 6a^3 + 12a^2 - 6a - 2a - 2a^3 + 12a^2 - 24a + 12 + 6\n\\]\n\\[\n= a^4 - 8a^3 + 24a^2 - 32a + 18\n\\]\n\nWe are given:\n\\[\na \\oplus (a \\oplus (a \\oplus a)) = a\n\\]\nSo,\n\\[\na^4 - 8a^3 + 24a^2 - 32a + 18 = a\n\\]\nSubtract \\( a \\) from both sides:\n\\[\na^4 - 8a^3 + 24a^2 - 33a + 18 = 0 \\tag{1}\n\\]\n\n---\n\n### Step 4. Solve the Polynomial Equation\n\nWe solve:\n\\[\na^4 - 8a^3 + 24a^2 - 33a + 18 = 0\n\\]\n\nTry \\( a = 2 \\):\n\\[\n16 - 64 + 96 - 66 + 18 = 0 \\quad \\Rightarrow \\quad a = 2 \\text{ is a root}\n\\]\n\nTry \\( a = 3 \\):\n\\[\n81 - 216 + 216 - 99 + 18 = 0 \\quad \\Rightarrow \\quad a = 3 \\text{ is a root}\n\\]\n\nFactor out \\( (a - 2)(a - 3) = a^2 - 5a + 6 \\). Divide the polynomial:\n\\[\n(a^4 - 8a^3 + 24a^2 - 33a + 18) \\div (a^2 - 5a + 6) = a^2 - 3a + 3\n\\]\n\nSo the factorization is:\n\\[\n(a^2 - 5a + 6)(a^2 - 3a + 3) = 0\n\\]\n\nSolve each quadratic:\n\n- \\( a^2 - 5a + 6 = 0 \\Rightarrow a = 2, 3 \\)\n- \\( a^2 - 3a + 3 = 0 \\Rightarrow a = \\frac{3 \\pm \\sqrt{9 - 12}}{2} = \\frac{3 \\pm i\\sqrt{3}}{2} \\)\n\n---\n\n### Step 5. Final Answer\n\nThe complex numbers \\( a \\) satisfying the equation are:\n\\[\n\\boxed{2}, \\quad \\boxed{3}, \\quad \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}\n\\]""": "\\boxed{2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}}",
    """### Step 1. Rewrite the Operation\n\nThe operation is defined as:\n\n\\[\nx \\oplus y = xy - 2x - 2y + 6\n\\]\n\nNotice that:\n\n\\[\n(x - 2)(y - 2) = xy - 2x - 2y + 4\n\\]\n\nSo we can write:\n\n\\[\nx \\oplus y = (x - 2)(y - 2) + 2\n\\]\n\nLet:\n\n\\[\n\\varphi(x) = x - 2\n\\]\n\nThen:\n\n\\[\n\\varphi(x \\oplus y) = (x \\oplus y) - 2 = (x - 2)(y - 2) = \\varphi(x) \\cdot \\varphi(y)\n\\]\n\nThis shows that the operation \\(\\oplus\\) is equivalent to multiplication under the transformation \\(\\varphi\\).\n\n---\n\n### Step 2. Change of Variable\n\nLet:\n\n\\[\nb = a - 2 \\quad \\Rightarrow \\quad a = b + 2\n\\]\n\nThen:\n\n\\[\na \\oplus a = (b + 2) \\oplus (b + 2) = b^2 + 2\n\\]\n\nNow compute:\n\n\\[\na \\oplus (a \\oplus a) = a \\oplus (b^2 + 2) = b^3 + 2\n\\]\n\nAnd:\n\n\\[\na \\oplus (a \\oplus (a \\oplus a)) = a \\oplus (b^3 + 2) = b^4 + 2\n\\]\n\nWe are given:\n\n\\[\na \\oplus (a \\oplus (a \\oplus a)) = a\n\\]\n\nSubstituting:\n\n\\[\nb^4 + 2 = b + 2 \\quad \\Rightarrow \\quad b^4 = b\n\\]\n\n---\n\n### Step 3. Solve the Equation\n\n\\[\nb^4 - b = 0 \\quad \\Rightarrow \\quad b(b^3 - 1) = 0\n\\]\n\nSo:\n\n- \\( b = 0 \\)\n- \\( b^3 = 1 \\Rightarrow b = 1, \\omega, \\omega^2 \\), where \\(\\omega = e^{2\\pi i/3} = \\frac{-1 + i\\sqrt{3}}{2}\\) and \\(\\omega^2 = \\frac{-1 - i\\sqrt{3}}{2}\\)\n\n---\n\n### Step 4. Convert Back to \\(a\\)\n\nSince \\(a = b + 2\\), we get:\n\n- \\( b = 0 \\Rightarrow a = 2 \\)\n- \\( b = 1 \\Rightarrow a = 3 \\)\n- \\( b = \\omega \\Rightarrow a = 2 + \\omega = \\frac{3 + i\\sqrt{3}}{2} \\)\n- \\( b = \\omega^2 \\Rightarrow a = 2 + \\omega^2 = \\frac{3 - i\\sqrt{3}}{2} \\)\n\n---\n\n### \u2705 Final Answer\n\n\\[\n\\boxed{2},\\quad \\boxed{3},\\quad \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}\n\\]""": "\\boxed{2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}}",
    """### Step 1. Simplify the Operation\n\nThe operation is defined as:\n\n\\[\nx \\oplus y = xy - 2x - 2y + 6\n\\]\n\nNotice that:\n\n\\[\n(x - 2)(y - 2) = xy - 2x - 2y + 4\n\\]\n\nSo we can rewrite:\n\n\\[\nx \\oplus y = (x - 2)(y - 2) + 2\n\\]\n\n---\n\n### Step 2. Compute the Nested Expression\n\nWe want to find all complex numbers \\( a \\) such that:\n\n\\[\na \\oplus (a \\oplus (a \\oplus a)) = a\n\\]\n\nLet\u2019s compute step by step:\n\n- **First step**:  \n  \\[\n  a \\oplus a = (a - 2)^2 + 2\n  \\]\n\n- **Second step**:  \n  \\[\n  a \\oplus (a \\oplus a) = (a - 2)^3 + 2\n  \\]\n\n- **Third step**:  \n  \\[\n  a \\oplus (a \\oplus (a \\oplus a)) = (a - 2)^4 + 2\n  \\]\n\nSo the equation becomes:\n\n\\[\n(a - 2)^4 + 2 = a\n\\]\n\n---\n\n### Step 3. Solve the Equation\n\nSubtract 2 from both sides:\n\n\\[\n(a - 2)^4 = a - 2\n\\]\n\nLet \\( b = a - 2 \\). Then:\n\n\\[\nb^4 = b \\quad \\Rightarrow \\quad b^4 - b = 0 \\quad \\Rightarrow \\quad b(b^3 - 1) = 0\n\\]\n\nSo:\n\n- \\( b = 0 \\)\n- \\( b^3 = 1 \\Rightarrow b = 1, \\omega, \\omega^2 \\), where \\( \\omega = \\frac{-1 + i\\sqrt{3}}{2} \\), \\( \\omega^2 = \\frac{-1 - i\\sqrt{3}}{2} \\)\n\nSince \\( a = b + 2 \\), we get:\n\n- \\( a = 2 \\)\n- \\( a = 3 \\)\n- \\( a = 2 + \\omega = \\frac{3 + i\\sqrt{3}}{2} \\)\n- \\( a = 2 + \\omega^2 = \\frac{3 - i\\sqrt{3}}{2} \\)\n\n---\n\n### \u2705 Final Answer\n\n\\[\n\\boxed{2}, \\quad \\boxed{3}, \\quad \\boxed{\\frac{3 \\pm i\\sqrt{3}}{2}}\n\\]""": "\\boxed{2,3,\\frac{3 + i\\sqrt{3}}{2}, \\frac{3 - i\\sqrt{3}}{2}}",
}


@total_ordering
class WarningType(Enum):
    """An enumeration for warning levels."""

    NONE = 0
    MINOR = 1
    POSSIBLE = 2
    MAJOR = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return self.value < other


def latex2sympy_fixed(latex: str):
    # if _integer is present, replace it with _{integer} for any integer
    latex = re.sub(r"_([0-9]+)", r"_{\1}", latex)
    latex_parsed = parse_latex(latex)
    # replace constants like pi and e with their numerical value
    known_constants = {"pi": sympy.pi, "e": sympy.E, "I": 1j, "i": 1j}

    # Replace any symbol in expr that is in our known_constants dictionary.
    expr = latex_parsed.xreplace(
        {s: known_constants[s.name] for s in latex_parsed.free_symbols if s.name in known_constants}
    )
    return expr


def remove_inner_boxed(match: str):
    """Removes inner `\boxed` or `\fbox` commands from a string.

    Args:
        match (str): The string to process.

    Returns:
        str: The string with inner `\boxed` or `\fbox` commands removed.
    """
    pattern = r"(\\boxed|\\fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, match))
    if not matches:
        return match
    for m in matches:
        match = match.replace(m.group(0), m.group(2))
    return match


def find_last_boxed_content(text: str, list_answer: bool = False) -> Optional[str]:
    """Finds the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the content of the last `\boxed` or `\fbox` command
            and a warning level.
    """
    pattern = r"(boxed|fbox)\{((?:[^{}]|\{(?2)\})*)\}"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.NONE

    if len(matches) > 1 and list_answer:
        # find all boxed content on the same line (no \n in between) as the last boxed
        split_text = text.split("\n")
        for i in range(len(split_text) - 1, -1, -1):
            matches_line = list(regex.finditer(pattern, split_text[i]))
            if len(matches_line) > 0:
                returned_boxed = ",".join([match.group(2) for match in matches_line])
                return remove_inner_boxed(returned_boxed), WarningType.POSSIBLE

    last_match = remove_inner_boxed(matches[-1].group(2))
    return last_match, WarningType.NONE


def extract_boxed_answer(text: str, list_answer: bool = False) -> Optional[str]:
    """Extracts the content of the last `\boxed` or `\fbox` command in a string.

    Args:
        text (str): The string to search.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the content of the last `\boxed` or `\fbox` command
            and a warning level.
    """
    answer, warning = find_last_boxed_content(text, list_answer)
    if answer is not None:
        return answer, warning
    else:
        return None, warning


def replace_and_or(s: str) -> str:
    """Replaces 'and' or 'or' with commas in a string.

    1) If 'and/or' (or their \text{} forms) is NOT right next to a comma
       (ignoring spaces) -> replace it by a single ','.
    2) Otherwise (comma already on at least one side) -> delete it.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string.
    """
    TOKEN = re.compile(
        r"""
        (?:\\text\s*\{\s*)?      # optional '\text{' and any leading blanks
        (and|or)                 # the word itself
        (?:\s*\})?               # optional closing '}' with any blanks
        """,
        re.I | re.VERBOSE,
    )
    # We build a fresh output string piece-by-piece so that each check
    # uses the **current** comma layout, not the one from the original text.
    out, idx = [], 0
    for m in TOKEN.finditer(s):
        start, end = m.span()
        # copy text *before* the token
        out.append(s[idx:start])

        # look to the left of the token, skipping blanks
        j = start - 1
        while j >= 0 and s[j].isspace():
            j -= 1
        comma_left = j >= 0 and s[j] == ","

        # look to the right of the token, skipping blanks
        k = end
        while k < len(s) and s[k].isspace():
            k += 1
        comma_right = k < len(s) and s[k] == ","

        # choose replacement
        out.append("" if (comma_left or comma_right) else ",")
        idx = end  # advance cursor

    out.append(s[idx:])  # tail of string
    return "".join(out)


def extract_boxed_answer_parse(text: str, parse: bool = True, list_answer: bool = False) -> Optional[int]:
    """Extracts and parses the content of the last `\boxed` or `\fbox` command.

    Args:
        text (str): The string to search.
        parse (bool, optional): Whether to parse the answer. Defaults to True.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    answer, warning = extract_boxed_answer(text, list_answer)
    if answer is not None:
        if answer.count("=") > 1:
            warning = max(warning, WarningType.MAJOR)  # this is a major warning, we should not have more than one "="
        try:
            return sympy.Integer(int(answer)), warning
        except:  # noqa: E722
            # logger.info(f"Could not parse answer {answer} as integer")
            if parse:
                parsed_answer, warning = parse_answer(answer, list_answer=list_answer)
                return parsed_answer, warning
            return answer, warning
    return None, WarningType.MAJOR


def extract_last_integer(text: str) -> Optional[int]:
    """Extracts the last integer from a string.

    Args:
        text (str): The string to search.

    Returns:
        tuple: A tuple containing the last integer and a warning level.
    """
    pattern = r"\b\d+\b"
    matches = list(regex.finditer(pattern, text))
    if not matches:
        return None, WarningType.MAJOR
    try:
        return int(matches[-1].group()), WarningType.MAJOR
    except Exception as e:
        logger.warning(f"Error extracting last integer: {e}")
        return None, WarningType.MAJOR


def extract_answer(text: str, strict_parsing: bool = True, parse: bool = True, list_answer: bool = False):
    """Extracts and parses the final answer from a string.

    Args:
        text (str): The string to search.
        strict_parsing (bool, optional): Whether to use strict parsing. Defaults to True.
        parse (bool, optional): Whether to parse the answer. Defaults to True.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    if text is None or len(text) == 0:
        return None, WarningType.MAJOR
    warning_old = WarningType.NONE
    if text in complete_mapper:
        text = complete_mapper[text]
        warning_old = WarningType.MAJOR
    text, warning = replace_unicode(text)
    warning = max(warning, warning_old)
    answer, warning_new = extract_boxed_answer_parse(text, parse, list_answer)
    if isinstance(answer, AnswerList) and len(answer.answers) == 1:
        answer = answer.answers[0]
    warning = max(warning, warning_new)
    if answer is not None or strict_parsing:
        return answer, warning

    return extract_last_integer(text)

def parse_answer(s: str, primitive_type: type = None, list_answer: bool = False):
    """Parses a string into a mathematical expression.

    Args:
        s (str): The string to parse.
        primitive_type (type, optional): The primitive type to parse into. Defaults to None.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        tuple: A tuple containing the parsed answer and a warning level.
    """
    warning = WarningType.NONE
    if s in manual_mapper:
        logger.warning(f"Applying manual parsing to {s}")
        s = manual_mapper[s]
        warning = WarningType.MAJOR
    s = remove_invalid_characters(s)
    s = remove_outer_brackets(normalize_string(s, list_answer))
    # s = insert_implicit_mul(s)
    output, warning_new = ParseList.parse("(" + s + ")", primitive_type=primitive_type)
    warning = max(warning, warning_new)
    if output is None:
        logger.warning(f"Could not parse {s}, returning None")
        return None, max(warning, WarningType.MAJOR)
    if len(output) == 1:
        output = output[0]

    if isinstance(output, list) or isinstance(output, tuple):
        output = AnswerList(output)
    return output, warning


def normalize_string(s, list_answer=False):
    """Normalizes a string for parsing.

    Args:
        s (str): The string to normalize.
        list_answer (bool, optional): Whether to expect a list of answers. Defaults to False.

    Returns:
        str: The normalized string.
    """
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\Bigl", "").replace(r"\Bigr", "")
    s = s.replace(r"\bigl", "").replace(r"\bigr", "")
    s = s.replace(r"\Big", "").replace(r"\big", "").replace(r"\Large", "").replace(r"\large", "")
    s = remove_aligns(s)
    s = s.replace("[", "(")
    s = s.replace("]", ")")
    s = s.replace("\\{", "(")  # sets will be converted to lists
    s = s.replace("\\}", ")")  # sets will be converted to lists
    s = s.replace("$", "")
    s = s.replace("\\ ", " ")
    # remove hline and vline
    s = s.replace(r"\hline", "")
    s = s.replace(r"\vline", "")
    s = s.replace(r"\quad", " ")
    s = s.replace("−", "-")
    s = s.replace("–", "-")
    s = s.replace("·", " \\cdot ")
    s = s.replace("^\\circ", " ")
    s = s.replace("^{\\circ}", " ")
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\(", "(")
    s = s.replace("\\)", ")")
    s = s.replace("{,}", "")  # o4-mini does this
    # remove \\begin{anything} and \\end{anything}
    if s.endswith("."):
        s = s[:-1]

    if list_answer and s is not None:
        s = replace_and_or(s)

    if not list_answer:
        # replace something of the type integer,integer with integerinteger
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
        s = s.replace("{,}", "")
    if list_answer:
        s = s.replace(";", ",")
        s = s.replace("{,}", ",")
    # if we see \sqrt 123ea pi\frac -> \sqrt{123ea}pi\frac
    if "\\sqrt " in s:
        s = re.sub(r"\\sqrt\s*([^\s{}]*)", r"\\sqrt{\1}", s)
    # remove everything that appears within \text{...}
    s = re.sub(r"\\text\{.*?\}", "", s)
    # replace \mathrm{...} with ...
    s = re.sub(r"\\mathrm\{(.*?)\}", r" \1 ", s)

    s = s.replace("F_{30}", "832040")  # Fibonacci number present in one problem
    if "=" in s:
        s = s.split("=")[-1]
    if r"\in" in s and list_answer:
        s = s.split(r"\in")[-1]

    if "\\approx" in s:
        s = s.split("\\approx")[0]
        if s.endswith("("):  # in case it was put in brackets
            s = s[:-1]
    return strip(s)


def remove_outer_brackets(s):
    """Removes the outermost matching brackets from the string if they encompass the entire string.

    Parameters:
    s (str): The input string potentially wrapped with brackets.

    Returns:
    str: The string with the outermost brackets removed if they match and encompass the entire string.
    """
    while True:
        if not s:
            return s
        opening = s[0]
        closing = s[-1]

        if opening == "(" and closing == ")":
            count = 0
            matched = True
            for i, char in enumerate(s):
                if char == opening:
                    count += 1
                elif char == closing:
                    count -= 1
                if count == 0 and i != len(s) - 1:
                    matched = False
                    break

            if matched:
                s = s[1:-1]
                continue
        break

    return s


def remove_aligns(s: str) -> str:
    """Removes `\\begin{align}` and `\\end{align}` environments from a string.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string.
    """
    # This pattern captures:
    #   \begin{align followed by any non-} characters (like align*, alignat, etc.)
    #   then any content (non-greedily) up to
    #   \\end{align...} with the same "align" prefix
    pattern = r"\\begin{align[^}]*}(.*?)\\end{align[^}]*}"

    # Use a callback to remove '&' from the matched group before returning it
    return re.sub(pattern, lambda m: m.group(1).replace("&", "").replace("\\\\", ""), s, flags=re.DOTALL)


def replace_unicode(text: str) -> str:
    """Replaces unicode characters with their LaTeX equivalents.

    Args:
        text (str): The string to process.

    Returns:
        tuple: A tuple containing the processed string and a warning level.
    """
    text_old = text
    text = text.replace("\u23a7", r"\boxed{")
    text = text.replace("\u23ab", r"}")
    text = text.replace("\n\u2502", r"\boxed{")
    text = text.replace("\u2502", r"}")
    text = text.replace("\n\u2503", r"\boxed{")
    text = text.replace("\u2503", r"}")
    text = text.replace("\n\uf8f0", r"\boxed{")
    text = text.replace("\uf8fb", r"}")
    warning = WarningType.NONE if text == text_old else WarningType.POSSIBLE
    text = text.replace("\u221a", r"\sqrt")  # these ones are for sure fine, no warning necessary
    text = text.replace("\u00d7", r"\cdot")
    text = text.replace("\u202f", r" ")
    text = text.replace("\u2212", "-")
    text = text.replace("\u03c0", r"\pi")
    return text, warning


def remove_invalid_characters(text):
    """Removes invalid characters from a string.

    Args:
        text (str): The string to process.

    Returns:
        str: The processed string.
    """
    text = re.sub(r"\\;", "", text)
    text = re.sub(r"\\:", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\!", "", text)
    return text


def strip(s: str):
    s = s.strip()
    # be careful with this, it can also remove the "\" in "\begin" if just done with strip
    while s.startswith(r"\n"):
        s = s[2:]
    while s.endswith(r"\n"):
        s = s[:-2]
    while s.startswith("\\ "):
        s = s[2:]
    # if s starts with any thing of the form \\\ and then a bracket, or \\\n and then a bracket, remove it
    while re.match(r"\\{2,}\n?\(", s):
        s = s[3:]
    return s


def split_multiletter_symbols(expr):
    reps = {}
    for s in list(expr.free_symbols):
        name = s.name
        if name.isalpha() and len(name) > 1 and not all(ch in "ABCDE" for ch in name):
            reps[s] = Mul(*[Symbol(ch) for ch in name])
    return expr.xreplace(reps)

def check_answers(ans1, ans2):
    """Checks if two answers are equal.

    Args:
        ans1: The first answer.
        ans2: The second answer.

    Returns:
        bool: True if the answers are equal, False otherwise.
    """
    if ans1 is None or ans2 is None:
        return False
    if (type(ans1) in [list, AnswerList]) != (type(ans2) in [list, AnswerList]):
        return False
    try:
        if not (hasattr(ans1, "equals") and callable(ans1.equals)) or not (
            hasattr(ans2, "equals") and callable(ans2.equals)
        ):
            # do approximate equal here
            if isinstance(ans1, str) or isinstance(ans2, str):
                # sympy check equality
                return bool(ans1 == ans2)
            err = abs(N(ans1 - ans2))
            if err < 1e-10 and err / max(abs(N(ans1)), abs(N(ans2))) < 1e-10:
                return True
            return False
        
        if not isinstance(ans1, AnswerList):
            ans1 = split_multiletter_symbols(ans1)
        if not isinstance(ans2, AnswerList):
            ans2 = split_multiletter_symbols(ans2)
        return bool(ans1.equals(ans2))
    except Exception as e:
        logger.warning(f"Error comparing answers {ans1} and {ans2}: {e}")
        return False


class AnswerList:
    """A class for representing a list of answers."""

    def __init__(self, answers: list[Any]):
        """Initializes the AnswerList.

        Args:
            answers (list[Any]): A list of answers.
        """
        if not isinstance(answers, list) and not isinstance(answers, tuple):
            raise ValueError(f"Expected passed answers to be list or tuple, received {type(answers)}")

        valid_answers = []
        for answer in answers:
            if bool(re.search(r"\d", str(answer))):
                valid_answers.append(answer)
            else:
                logger.warning(f"Could not find any numbers in {answer}, removed from list")

        self.answers = list(valid_answers)

    def equals(self, other: list[Any]):
        """Checks if this AnswerList is equal to another list of answers.

        Args:
            other (list[Any]): The other list of answers.

        Returns:
            bool: True if the lists are equal, False otherwise.
        """
        if len(self.answers) != len(other):
            # logger.info(f"Lists {self.answers} and {other} do not have the same length.")
            return False

        match_ids = set()
        for ans1 in self.answers:
            match_found = False
            for i, ans2 in enumerate(other):
                if i not in match_ids and check_answers(ans1, ans2):
                    match_ids.add(i)
                    match_found = True
                    break
            if not match_found:
                # logger.info(f"Could not find a match for element {ans1} in {other}")
                return False
        return True

    def __str__(self):
        return "[" + ",".join([str(ans) for ans in self.answers]) + "]"

    def __len__(self):
        return len(self.answers)

    def __iter__(self):
        return iter(self.answers)


class ParseObject:
    """A base class for parsing objects."""

    @classmethod
    def is_at_start(cls, string):
        """Checks if the object is at the start of a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is at the start of the string, False otherwise.
        """
        return False

    @classmethod
    def is_complete(cls, string):
        """Checks if the object is complete in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is complete in the string, False otherwise.
        """
        return string.count("{") == string.count("}") and string.count("(") == string.count(")")

    @classmethod
    def is_finished(cls, string):
        """Checks if the object is finished in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is finished in the string, False otherwise.
        """
        return True

    @classmethod
    def parse(cls, string):
        """Parses a string into an object.

        Args:
            string (str): The string to parse.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class ParsePrimitive(ParseObject):
    """A class for parsing primitive types."""

    @classmethod
    def parse(cls, string, primitive_type):
        """Parses a string into a primitive type.

        Args:
            string (str): The string to parse.
            primitive_type (type): The primitive type to parse into.

        Returns:
            tuple: A tuple containing the parsed primitive and a warning level.
        """
        warning = WarningType.NONE
        # Integer
        if string.isdigit():
            if primitive_type == Fraction:
                return Fraction(int(string), 1)
            return int(string), warning
        # Float
        try:
            float_string = float(string)
            if int(float_string) == float_string:
                if primitive_type == Fraction:
                    return Fraction(int(float_string), 1)
                return int(float_string), warning
            return float_string, warning
        except ValueError:
            # logger.info(f"Couldn't configure floating point to fraction for {string}")
            pass
        # Expression
        if bool(re.search(r"sqrt(\d+)", string)):
            string = re.sub(r"sqrt(\d+)", r"sqrt{\1}", string)
        if bool(re.search(r"frac(\d)", string)):
            string = re.sub(r"frac(\d)", r"frac{\1}", string)
        try:
            latex_str = string
            for _ in range(5):
                init_str = latex_str
                latex_str = re.sub(r"\\*(?:dfrac|tfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", latex_str)
                latex_str = re.sub(r"\\*binom\{([^{}]*)\}\{([^{}]*)\}", r"binomial(\1, \2)", latex_str)
                latex_str = re.sub(r"\\*sqrt\[(.*?)\]\{(.*?)\}", r"(\2)**(1/(\1))", latex_str)
                latex_str = re.sub(r"\\*sqrt\{(.*?)\}", r"(\1)**(1/2)", latex_str)

                latex_str = latex_str.replace("^", "**")
                latex_str = latex_str.replace("\\cdot", "*").replace("\\times", "*")
                latex_str = latex_str.replace("\\pi", " pi ").replace("\\e", " E ").replace("\\i", " I ")
                latex_str = re.sub(r"\bi\b", "I", latex_str)
                if init_str == latex_str:
                    break

            for _ in range(5):

                init_str = latex_str
                latex_str = re.sub(r"\{(\d+)\}", r"(\1)", latex_str)
                latex_str = re.sub(r"\\*(?:dfrac|tfrac|frac)\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", latex_str)
                latex_str = re.sub(r"\\*binom\{([^{}]*)\}\{([^{}]*)\}", r"binomial(\1, \2)", latex_str)
                latex_str = re.sub(r"\\*sqrt\[(.*?)\]\{(.*?)\}", r"(\2)**(1/(\1))", latex_str)
                latex_str = re.sub(r"\\*sqrt\{(.*?)\}", r"(\1)**(1/2)", latex_str)

                latex_str = latex_str.replace("^", "**")
                latex_str = latex_str.replace("\\cdot", "*").replace("\\times", "*")
                latex_str = latex_str.replace("\\pi", " pi ").replace("\\e", " E ").replace("\\i", " I ")
                latex_str = re.sub(r"\bi\b", "I", latex_str)
                if init_str == latex_str:
                    break

            # Handle implcit multiplication
            latex_str = re.sub(r"(\d|(?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))\(", r"\1*(", latex_str)
            latex_str = re.sub(r"\)(\d|(?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))", r")*\1", latex_str)
            latex_str = re.sub(r"(?<=\d)((?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))", r"*\1", latex_str)
            latex_str = re.sub(r"((?<![a-zA-Z])[a-zA-Z]{1,2}(?![a-zA-Z]))(?=\d)", r"\1*", latex_str)
            latex_str = re.sub(r"\{([^{}]*)\}", lambda m: "[" + m.group(1).replace(",", ", ") + "]", latex_str)

            if latex_str == "None":
                string = sympy.core.symbol.Symbol("None")
            else:
                string = sympy.sympify(
                    latex_str,
                    locals={"binomial": sympy.binomial, "pi": sympy.pi, "E": sympy.E, "e": sympy.E, "I": sympy.I},
                )
        except Exception as e:
            try:
                string_no_eq = string
                if "=" in string_no_eq:
                    # rfind is used to remove the last occurence of "="
                    string_no_eq = string_no_eq[string_no_eq.rfind("=") + 1 :]
                output_val = latex2sympy_fixed(string_no_eq)
                # print complex and real part separately

                try:
                    float_val = float(N(output_val, 101))
                    if float_val.is_integer() or float("inf") == float_val or float("-inf") == float_val:
                        return int(N(latex2sympy_fixed(string_no_eq), 50001)), warning  # important for large ints
                    return float_val, warning
                except:  # noqa: E722
                    try:
                        complex_val = complex(N(output_val, 101))
                        return complex_val, warning
                    except:  # noqa: E722
                        return output_val, warning
            except Exception as e:
                logger.warning(f"Error: Custom parsing error {e}, {string_no_eq}")
                warning = max(warning, WarningType.MAJOR)
                return None, warning

        return string, warning

    @classmethod
    def is_at_start(cls, string):
        return True


class ParseList(ParseObject):
    """A class for parsing lists."""

    @classmethod
    def is_at_start(cls, string):
        """Checks if the object is at the start of a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is at the start of the string, False otherwise.
        """
        return string.startswith(r"(")

    @classmethod
    def is_finished(cls, string):
        """Checks if the object is finished in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is finished in the string, False otherwise.
        """
        # safe condition for finishing a list
        return string.strip().strip(",").endswith(")")

    @classmethod
    def is_complete(cls, string):
        """Checks if the object is complete in a string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the object is complete in the string, False otherwise.
        """
        return string.count("(") == string.count(")")

    @classmethod
    def never_zero_count(cls, string):
        """Checks if the parenthesis count never reaches zero before the end of the string.

        Args:
            string (str): The string to check.

        Returns:
            bool: True if the parenthesis count never reaches zero, False otherwise.
        """
        # says wheter count "(" - count ")" for every string[:i] is never zero
        count = 0
        ever_zero = False
        for char in string:
            if char == "(":
                count += 1
            if char == ")":
                count -= 1
            if count == 0:
                ever_zero = True
        return not ever_zero

    @classmethod
    def parse(cls, string, delimiter=[r"\n", ","], primitive_type=None, depth=0):
        """Parses a string into a list.

        Args:
            string (str): The string to parse.
            delimiter (list[str], optional): The delimiter to use. Defaults to [r"\n", ","].
            primitive_type (type, optional): The primitive type to parse into. Defaults to None.
            depth (int, optional): The recursion depth. Defaults to 0.

        Returns:
            tuple: A tuple containing the parsed list and a warning level.
        """
        if isinstance(delimiter, str):
            delimiter = [delimiter]
        output = []
        if not string.startswith("("):
            return None
        string = string.strip().strip(",")
        if cls.never_zero_count(string[:-1]):
            string = string[1:-1]
        string = strip(string)
        used_delim = delimiter[0]
        for delim in delimiter:
            if delim in string:
                comma_separated = string.split(delim)
                used_delim = delim
                break
        warning = WarningType.NONE
        while len(string) > 0:
            previous_string = string
            comma_separated = string.split(used_delim)
            allowed_objects = [ParseList, ParsePrimitive]
            if depth > 50:
                allowed_objects = [ParsePrimitive]
            for obj in allowed_objects:
                if obj.is_at_start(strip(string)):
                    current_index = 1
                    while not obj.is_complete(
                        strip(used_delim.join(comma_separated[:current_index]))
                    ) or not obj.is_finished(strip(used_delim.join(comma_separated[:current_index]))):
                        current_index += 1
                        if current_index >= len(comma_separated):
                            break
                    if not obj.is_complete(
                        strip(used_delim.join(comma_separated[:current_index]))
                    ) or not obj.is_finished(strip(used_delim.join(comma_separated[:current_index]))):
                        continue

                    if obj == ParseList:
                        parsed, new_warning = obj.parse(
                            strip(used_delim.join(comma_separated[:current_index])),
                            primitive_type=primitive_type,
                            depth=depth + 1,
                        )
                    else:
                        parsed, new_warning = obj.parse(
                            strip(used_delim.join(comma_separated[:current_index])), primitive_type=primitive_type
                        )
                    warning = max(warning, new_warning)
                    output.append(parsed)
                    string = strip(used_delim.join(comma_separated[current_index:]))
                    break
            if previous_string == string:
                if depth > 50:
                    logger.error(f"Response {string} reached depth > 50")
                    raise ValueError(f"Failed to parse '{string}'")
                return None, WarningType.MAJOR
        return output, warning