
require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
require 'table_utils'

nngraph.setDebug(true)



criterion = nn.ClassNLLCriterion()
softmax = nn.LogSoftMax()
x = torch.zeros(3,2)

y = torch.ones(3,1)
y[1][1] = 1
y[2][1] = 2
y[3][1] = 1

z = softmax:forward(x)
loss = criterion:forward(z, y)

print(loss)

