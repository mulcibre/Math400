#   Samuel Gluss
#   5-10-2016
#   Emily Conway

#   define output
set terminal pngcairo size 800,550 enhanced font 'Verdana,10'
set output "a5p3output.png"

#   set title, labels, key position
set title "Piecewise Approximation with Legendre Polynomial Basis Functions"
set xlabel "<-- (x) -->"
set ylabel "<-- (y) -->"
set key top left title 'Legend' box 3

#   show axes aligned on origin
set zeroaxis

#   set domain/range
    set xr [-1.5:1.5]
    set yr [-0.5:1.0]

#   define piecewise function
a(x) =  (x >= -1.0 && x <= -0.5) ? 0 : \
        (x > -0.5 && x < 0.5) ? 0.5 : \
        (x >= 0.5 && x <= 1.0) ? 0 : \
        1/0

#   define coefficients
a0 = (1.0/4)
a1 = 0
a2 = (-13.0/32)
a3 = 0
a4 = (135.0/512)
        
#   define Legendre Polynomial functions
p0(x) = 1
p1(x) = x 
p2(x) = (1.0 / 2) * (3 * x**2 - 1)
p3(x) = (1.0 / 2) * (5 * x**3 - 3 * x)
p4(x) = (1.0 / 8) * (35 * x**4 - 30 * x**2 + 3)

#   define least squares approximating function
f1(x) = a0 * p0(x)
f2(x) = a0 * p0(x) + a1*p1(x)
f3(x) = a0 * p0(x) + a1*p1(x) + a2*p2(x)
f4(x) = a0 * p0(x) + a1*p1(x) + a2*p2(x) + a3*p3(x)
f5(x) = a0 * p0(x) + a1*p1(x) + a2*p2(x) + a3*p3(x) + a4*p4(x)

plot a(x) , f1(x), f2(x), f3(x), f4(x), f5(x)