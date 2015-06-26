
require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'
require 'table_utils'

nngraph.setDebug(true)



l = {'1', 'asdf', '1234'}
s = table.concat(l, ' ')

print(s)