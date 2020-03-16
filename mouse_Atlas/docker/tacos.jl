module Tacos
using PyCall
using Statistics

pickle = pyimport("pickle")

export estimate_r2, load_pickle

estimate_r2 = function(f, params, x_data, y_data)
	residuals = [y - f(x, params)[1] for (x, y) in zip(x_data,y_data)]
	ss_res = sum([r^2 for r in residuals])
	ss_tot = sum([y^2 for y in [y1 - mean(y_data) for y1 in y_data]])
	r_squared = 1 - (ss_res / ss_tot)
	return r_squared
end

function load_pickle(filename)
	@pywith pybuiltin("open")(filename,"rb") as f begin
	   data = nothing
	   data = pickle.load(f)
	   return data
	end
end

end