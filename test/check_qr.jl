using DelimitedFiles
using LinearAlgebra
using Dates

for fp in ["A.csv", "Q.csv", "R.csv"]
	println("$(fp) modified $(Dates.unix2datetime(mtime(fp)))")
end

A = readdlm("A.csv", ',');
Q = readdlm("Q.csv", ',');
R = readdlm("R.csv", ',');

println("|A-Q*R|/|A|=$(norm(A-Q*R)/norm(A))");
println("|I-Q'*Q|=$(norm(I-Q'*Q))");