module Mazzolini

using DataFrames
using Distributions
using Base.Threads

export run_mazzolini

function get_marginals(df)
    M = sum(df, dims = 1);
    f = sum(df, dims = 2);
    f = reshape(f/sum(f), (length(f)));
    return (M, f)
end

function run_mazzolini(df_data)
    df = convert(Array, df_data)[:,2:end]
    M, f = get_marginals(df)
    new_df = []
    W = length(f)
    for m in M
        dist = Multinomial(Int(round(m)), f)
        append!(new_df,reshape(rand(dist),(1,W)))
    end
    new_df = reshape(new_df,(W,1000))
    size(new_df)
    mazzolini_df = DataFrame([df_data[:,1] new_df])
    mazzolini_df = rename!(mazzolini_df, names(df_data))
    return mazzolini_df
end

function run_parallel_mazzolini(df_data)
	df = convert(Array, df_data)[:,2:end]
	M, f = get_marginals(df)
	c = Channel(length(M))
    new_df = []
    W = length(f)
	function sample(m)
		dist = Multinomial(Int(round(m)), f)
        put!(c,reshape(rand(dist),(1,W)))
	end
	
	function append()
		append!(new_df,take!(c))
	end
	
    @sync for m in M
		@async sample(m)
        @async append()
    end
	close(c)
    new_df = reshape(new_df,(W,1000))
    size(new_df)
    mazzolini_df = DataFrame([df_data[:,1] new_df])
    mazzolini_df = rename!(mazzolini_df, names(df_data))
    return mazzolini_df
end

end