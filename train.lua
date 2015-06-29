require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)

rnn_size = 100
vocab_size = 10002

--train data
function read_words(fn)
  fd = io.lines(fn)
  sentences = {}
  line = fd()

  while line do
    sentence = {}
    for _, word in pairs(string.split(line, " ")) do
        sentence[#sentence + 1] = word
    end
    sentences[#sentences + 1] = sentence
    line = fd()
  end
  return sentences
end

function convert2tensors(sentences)
  l = {}
  for _, sentence in pairs(sentences) do
    t = torch.zeros(1, #sentence)
    for i = 1, #sentence do 
      t[1][i] = sentence[i]
    end
    l[#l + 1] = t
  end
  return l  
end

sentences_ru = read_words('filtered_sentences_indexes_ru_rev')
sentences_en = read_words('filtered_sentences_indexes_en')

sentences_ru = convert2tensors(sentences_ru)
sentences_en = convert2tensors(sentences_en)

--print(sentences_ru)

assert(#sentences_en == #sentences_ru)
n_data = #sentences_en

--encoder
x = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(rnn_size, rnn_size)(x):annotate{name='i2h'}
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h):annotate{name='h2h'}
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

encoder = nn.gModule({x, prev_c, prev_h}, {next_c, next_h})


--decoder
x = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()

function new_input_sum()
    -- transforms input
    i2h            = nn.Linear(rnn_size, rnn_size)(x)
    -- transforms previous timestep's output
    h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
    return nn.CAddTable()({i2h, h2h})
end

in_gate          = nn.Sigmoid()(new_input_sum())
forget_gate      = nn.Sigmoid()(new_input_sum())
out_gate         = nn.Sigmoid()(new_input_sum())
in_transform     = nn.Tanh()(new_input_sum())

next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_transform})
})
next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

prediction = nn.Linear(rnn_size, vocab_size)(next_h)
prediction = nn.LogSoftMax()(prediction)

decoder = nn.gModule({x, prev_c, prev_h}, {next_c, next_h, prediction})


--embedding layer fed into encoder
embed_enc = Embedding(vocab_size, rnn_size)

--embedding layer fed into decoder
embed_dec = Embedding(vocab_size, rnn_size)

criterion = nn.ClassNLLCriterion()

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(embed_enc, embed_dec, encoder, decoder)
params:uniform(-0.08, 0.08)

seq_length = 30

-- make a bunch of clones, AFTER flattening, as that reallocates memory
embed_enc_clones = model_utils.clone_many_times(embed_enc, seq_length)
embed_dec_clones = model_utils.clone_many_times(embed_dec, seq_length)
encoder_clones = model_utils.clone_many_times(encoder, seq_length)
decoder_clones = model_utils.clone_many_times(decoder, seq_length)
criterion_clones = model_utils.clone_many_times(criterion, seq_length)


x_raw_enc = sentences_ru
x_raw_dec = sentences_en
iteration_counter = 1

-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    ------------------- forward pass -------------------
    lstm_c_enc = {[0]=torch.zeros(1, rnn_size)}
    lstm_h_enc = {[0]=torch.zeros(1, rnn_size)}
    x_enc_embedding = {}
        
    local loss = 0
    
    x_enc = x_raw_enc[iteration_counter]
    for t = 1, x_enc:size(2) - 1 do
      x_enc_embedding[t] = embed_enc_clones[t]:forward(x_enc[{{}, {t}}]:reshape(1))
      lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder_clones[t]:forward({x_enc_embedding[t], lstm_c_enc[t-1], lstm_h_enc[t-1]}))
    end
    
    lstm_c_dec = {[0]=torch.zeros(1, rnn_size)}
    lstm_h_dec = {[0]=lstm_h_enc[x_enc:size(2)-1]}
    x_dec_prediction = {}
    x_dec_embedding = {}
    
    x_dec = x_raw_dec[iteration_counter]
    x_dec_embedding[0] = embed_dec_clones[1]:forward(torch.Tensor(1):fill(vocab_size))
    for t = 1, x_dec:size(2) - 1 do 
      x_dec_embedding[t] = embed_dec_clones[t]:forward(x_dec[{{}, {t}}]:reshape(1))
      lstm_c_dec[t], lstm_h_dec[t], x_dec_prediction[t] = unpack(decoder_clones[t]:forward({x_dec_embedding[t-1], lstm_c_dec[t-1], lstm_h_dec[t-1]}))
      loss_x = criterion_clones[t]:forward(x_dec_prediction[t], x_dec[{{}, {t}}]:reshape(1))
      loss = loss + loss_x
      --print(loss_x)
            
    end
    loss = loss / (x_dec:size(2) - 1)

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_dec = {[x_dec:size(2) - 1] = torch.zeros(1, rnn_size)}
    dlstm_h_dec = {[x_dec:size(2) - 1] = torch.zeros(1, rnn_size)}
    dx_dec_prediction = {}
    dx_dec_embedding = {[x_dec:size(2) - 1] = torch.zeros(1, rnn_size)}
    dx_dec = {}
    dloss_x = {}
    
    for t = x_dec:size(2) - 1,1,-1 do
      dx_dec_prediction[t] = criterion_clones[t]:backward(x_dec_prediction[t], x_dec[{{}, {t}}]:reshape(1))
      dx_dec_embedding[t-1], dlstm_c_dec[t-1], dlstm_h_dec[t-1] = unpack(decoder_clones[t]:backward({x_dec_embedding[t-1], lstm_c_dec[t-1], lstm_h_dec[t-1]}, {dlstm_c_dec[t], dlstm_h_dec[t], dx_dec_prediction[t]}))
      dx_dec[t] = embed_dec_clones[t]:backward(x_dec[{{}, {t}}]:reshape(1), dx_dec_embedding[t])
    end
    
    dlstm_c_enc = {[x_enc:size(2) - 1] = torch.zeros(1, rnn_size)}
    dlstm_h_enc = {[x_enc:size(2) - 1] = dlstm_h_dec[0]}
    dx_enc_embedding = {}
    dx_enc = {}

        
    for t = x_enc:size(2) -1, 1, -1 do
      dx_enc_embedding[t], dlstm_c_enc[t-1], dlstm_h_enc[t-1] = unpack(encoder_clones[t]:backward({x_enc_embedding[t], lstm_c_enc[t-1], lstm_h_enc[t-1]}, {dlstm_c_enc[t], dlstm_h_enc[t]}))
      dx_enc[{{}, {t}}] = embed_enc_clones[t]:backward(x_enc[{{}, {t}}]:reshape(1), dx_enc_embedding[t])
    end
      
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)
    iteration_counter = iteration_counter + 1
    if iteration_counter > #x_raw_enc then 
      iteration_counter = 1
    end

    return loss, grad_params
end

optim_state = {learningRate = 1e-2}


for i = 1, 10000 do
  local _, loss = optim.adagrad(feval, params, optim_state)

  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end




