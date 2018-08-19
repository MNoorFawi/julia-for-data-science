Pkg.add("DataFrames")
Pkg.add("Query")
# .....

using DataFrames, Query, Knet, Gadfly,
  Cairo, Clustering, RDatasets

Uefa_Goalscorers = DataFrame(Player = ["Cristiano Ronaldo", "Lionel Messi", "Raul Gonzalez"], 
               Goals = [120, 100, 71])
Uefa_Goalscorers
describe(Uefa_Goalscorers)

Movies = dataset("ggplot2", "movies")
size(Movies)
names(Movies)
Movies2 = delete!(Movies, 
                  [:Budget, :Length, :R1, :R2, :R3, :R4, :R5,
                   :R6, :R7, :R8, :R9, :R10, :MPAA])
head(Movies2, 3)

Movies3 = stack(Movies2, 5:11)

head(Movies3, 3)
tail(Movies3, 3)

categorical!(Movies3, :variable)
rename!(Movies3, :variable => :Genre)
sort!(Movies3, :Year)

by(Movies3[Movies3[:value] .> 0, :], :Genre) do df
          DataFrame(MeanRating = mean(df[:Rating]), N = size(df, 1))
       end


Movies4 = @from i in Movies3 begin
    @where i.value > 0
    @select i
    @collect DataFrame
end

Movies4 = @from i in Movies4 begin
    @group i by i.Year into g
    @select {Year = g.key, Count = length(g)}
    @collect DataFrame
end


size(Movies4)
head(Movies4, 3)

plot(Movies4, x = :Year, y = :Count, Geom.line)

diamonds = dataset("ggplot2", "diamonds")
plot(diamonds, x = :Price, y = :Carat, Geom.hexbin)
plot(diamonds, x = :Price, color = :Cut, Geom.density)
plot(diamonds, x = :Cut, y = :Price, Geom.boxplot)

## Get the data
House = readtable("home_data-train .csv", separator = ',', header = false)
# remove the first two columns as they're not important
delete!(House, [:x1, :x2])
# examine correlation to select variables 
cor(convert(Array, House))
# Get x, y
excluded = [3, 7, 11, 14, 15, 16, 17, 18, 19, 20]
unnecessary = [Symbol("x$i") for i in excluded]
x = House[setdiff(names(House), unnecessary)]
# Convert x to a matrix
x = convert(Array, x)'
# Scale x
x = x ./ sum(x, 1)
y = House[:x3]'
y = [log10(i) for i in y] # log y 
## TRAIN THE MODEL
using Knet
predict(w, x) = w[1]*x .+ w[2] # linear regression equation ax + b
loss(w, x, y) = mean(abs2, y-predict(w, x)) # mean error
lossgradient = grad(loss) # lossgradient returns dw, the gradient of the loss 

function train(w, data; lr=.1) # lr learning rate
    for (x,y) in data
        dw = lossgradient(w, x, y)
	for i in 1:length(w)
	    w[i] -= lr * dw[i]
	end	    
    end
    return w
end

w = Any[0.1 * randn(1, 9), 0.0 ] # 9 variables

for i = 1:10; train(w, [(x, y)]); println(loss(w, x, y)); end
# 15.747867259203675
# 7.728037219389701
# .....
# 0.19558043796561383
# 0.1697502328566464

## First let's look at every variable cofficient and the intercept

w[1]
# 1×9 Array{Float64,2}:
#  -0.0188016  -0.148975  0.810052  0.0934263  0.149106  0.100036  -0.0526069  0.430641  2.39719

w[2]
# 3.6645523063680914

## Now the actual y

y
# 5.34616  5.73078  5.25527  5.78104  5.70757 ...

yhat = w[1] * x .+ w[2]
# 5.53536  5.38548  5.77452  5.41222  5.50971 ...

#### N.B. they're logged numbers to return them to actual do (10 ^ y)


Animals = dataset("MASS", "Animals")
using Clustering

feature_matrix = permutedims(convert(Array, Animals[:, 2:3]), 
                             [2, 1]) # to convert it to matrix
feature_matrix = collect(Missings.replace(feature_matrix, 0.0)) # to replace missings
model = kmeans(feature_matrix, 3) # 3 clusters           

## Plotting clusters
plot(Animals, x = :Brain, y = :Body, 
     color = categorical(model.assignments), label = :Species, 
     Geom.point, Geom.label, Scale.x_log10, Scale.y_log10)
