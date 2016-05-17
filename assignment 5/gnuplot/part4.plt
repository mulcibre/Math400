#   Samuel Gluss
#   5-10-2016
#   Emily Conway

#   define output
set terminal pngcairo size 800,550 enhanced font 'Verdana,10'
set output "a5p4output.png"

#   set title, labels, key position
set title "Piecewise approximation with Trig Basis Functions"
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
a0 = 1.0 / 2
a1 = 1.0 / pi
a2 = 0
a3 = -1.0 / (3 * pi)
a4 = 0
a5 = 1.0 / (5 * pi)

#   define Legendre Polynomial functions
p0(x) = 1.0 / 2
p1(x) = cos(pi * x) 
p2(x) = cos(2 * pi * x)
p3(x) = cos(3 * pi * x)
p4(x) = cos(4 * pi * x)
p5(x) = cos(5 * pi * x)

#   define least squares approximating function
f(x) = a0 * p0(x) + a1*p1(x) + a2*p2(x) + a3*p3(x) + a4*p4(x) + a5*p5(x)

plot a(x) , f(x)