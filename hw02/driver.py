from hw02_f19 import *


xor_wmats_231 = train_3_layer_nn(1000, X1, y_xor, build_231_nn) 
print(fit_3_layer_nn(X1[0], xor_wmats_231))
print(fit_3_layer_nn(X1[1], xor_wmats_231))
print(fit_3_layer_nn(X1[2], xor_wmats_231))
print(fit_3_layer_nn(X1[3], xor_wmats_231))

print("Xor 3 levels")

xor_wmats_2331 = train_4_layer_nn(1000, X1, y_xor, build_2331_nn) 
print(fit_4_layer_nn(X1[0], xor_wmats_2331))
print(fit_4_layer_nn(X1[1], xor_wmats_2331))
print(fit_4_layer_nn(X1[2], xor_wmats_2331))
print(fit_4_layer_nn(X1[3], xor_wmats_2331))

print("Xor 4 levels")

or_wmats_231 = train_3_layer_nn(1000, X1, y_or, build_231_nn) 
print(fit_3_layer_nn(X1[0], or_wmats_231))
print(fit_3_layer_nn(X1[1], or_wmats_231))
print(fit_3_layer_nn(X1[2], or_wmats_231))
print(fit_3_layer_nn(X1[3], or_wmats_231))

print("or 3 levels")

or_wmats_2331 = train_4_layer_nn(1000, X1, y_or, build_2331_nn) 
print(fit_4_layer_nn(X1[0], or_wmats_2331))
print(fit_4_layer_nn(X1[1], or_wmats_2331))
print(fit_4_layer_nn(X1[2], or_wmats_2331))
print(fit_4_layer_nn(X1[3], or_wmats_2331))

print("or 4 levels")

and_wmats_231 = train_3_layer_nn(1000, X1, y_and, build_231_nn) 
print(fit_3_layer_nn(X1[0], and_wmats_231))
print(fit_3_layer_nn(X1[1], and_wmats_231))
print(fit_3_layer_nn(X1[2], and_wmats_231))
print(fit_3_layer_nn(X1[3], and_wmats_231))

print("and 3 levels")

and_wmats_2331 = train_4_layer_nn(1000, X1, y_or, build_2331_nn) 
print(fit_4_layer_nn(X1[0], and_wmats_2331))
print(fit_4_layer_nn(X1[1], and_wmats_2331))
print(fit_4_layer_nn(X1[2], and_wmats_2331))
print(fit_4_layer_nn(X1[3], and_wmats_2331))

print("and 4 levels")

not_wmats_131 = train_3_layer_nn(1000, X2, y_not, build_121_nn) 
print(fit_3_layer_nn(X2[0], not_wmats_131))
print(fit_3_layer_nn(X2[1], not_wmats_131))
print("not 3 levels")

not_wmats_1331 = train_4_layer_nn(1000, X2, y_not, build_1331_nn) 
print(fit_4_layer_nn(X2[0], not_wmats_1331))
print(fit_4_layer_nn(X2[1], not_wmats_1331))
print("not 4 levels")

boolean_wmats_4o1 = train_3_layer_nn(1000, X4, bool_exp, build_4o1_nn) 
print(fit_3_layer_nn(X4[0], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[1], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[2], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[3], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[4], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[5], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[6], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[7], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[8], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[9], boolean_wmats_4o1))
print(fit_3_layer_nn(X4[10], boolean_wmats_4o1))
print("boolean 3 levels")
boolean_wmats_4o31 = train_4_layer_nn(1000, X4, bool_exp, build_4o31_nn) 
print("boolean 4 levels")
