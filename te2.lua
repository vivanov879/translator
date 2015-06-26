require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
local model_utils=require 'model_utils'
local mnist = require 'mnist'

nngraph.setDebug(true)

require 'table_utils'

l = {}
l[1] = {}
l[1][2] = '2'
l[1][3] = 3
l[2] = {}
l[4] = '4'
table.save(l , 't')
x = table.load('t')
print(x)
y, err = table.load('filtered_sentences_indexes_en')
print(y, err)
a = 1


