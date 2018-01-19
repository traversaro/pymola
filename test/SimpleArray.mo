model SimpleArray
    Real a[3] = {1.0, 2.0, 3.0};
    constant Real b[4] = {2.7, 3.7, 4.7, 5.7};
    Real c[3](each min = 0.0);
    Real[3] d;
    Real e[3];
    Real scalar_f = 1.3;
    Real g;
    constant Integer c_dim = 2;
    parameter Integer d_dim = 3;
equation
    // Array operators.
    c = a .+ b[1:d_dim].*e; // .+ is equal to + in this case

    // Calling a (scalar) function on an array maps the function to each element.
    d = sin(a ./ b[c_dim:4]);

    // Difference between .+ and +
    e = d .+ scalar_f; // Different shapes, so + is not allowed, only .+ is.

    // Sum.
    g = sum(c);

end ArrayExpressions;
