require 'mobdebug'.start()
require 'cutorch'
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'Embedding'
local model_utils=require 'model_utils'
require 'table_utils'
nngraph.setDebug(true)
require 'lstm'

opt = {}
opt.rnn_size = 40
opt.n_layers = 2
rnn_size = opt.rnn_size
n_layers = opt.n_layers
batch_size = 10

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

sentences_ru = read_words('filtered_sentences_indexes_ru_rev1')
sentences_en = read_words('filtered_sentences_indexes_en1')

function calc_max_sentence_len(sentences)
  local m = 1
  for _, sentence in pairs(sentences_en) do
    m = max(m, #sentence)
  end
  return m
end

max_sentence_len = max(calc_max_sentence_len(sentences_en), calc_max_sentence_len(sentences_ru))

sentences_ru = convert2tensors(sentences_ru)
sentences_en = convert2tensors(sentences_en)

--print(sentences_ru)

assert(#sentences_en == #sentences_ru)
n_data = #sentences_en

vocabulary_ru = table.load('vocabulary_ru')
vocabulary_en = table.load('vocabulary_en')
vocab_size = #vocabulary_ru
assert (#vocabulary_en == #vocabulary_ru)


--encoder
x = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()

m = make_lstm_network(opt)
next_x, next_c, next_h = m({x, prev_c, prev_h}):split(3)

encoder = (nn.gModule({x, prev_c, prev_h}, {next_x, next_c, next_h})):cuda()


--decoder
x = nn.Identity()()
prev_h = nn.Identity()()
prev_c = nn.Identity()()

m = make_lstm_network(opt)
next_x, next_c, next_h = m({x, prev_c, prev_h}):split(3)

prediction = nn.Linear(rnn_size, vocab_size)(next_x)
prediction = nn.LogSoftMax()(prediction)

decoder = (nn.gModule({x, prev_c, prev_h}, {next_c, next_h, prediction})):cuda()


--embedding layer fed into encoder
embed_enc = (Embedding(vocab_size, rnn_size)):cuda()

--embedding layer fed into decoder
embed_dec = (Embedding(vocab_size, rnn_size)):cuda()

criterion = (nn.ClassNLLCriterion()):cuda()

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
data_index = 1

function gen_batch()
  end_index = data_index + batch_size
  if end_index > n_data then
    end_index = n_data
    data_index = 1

  end
  start_index = end_index - data_index

  sentences = sentences_ru
  t = torch.zeros(batch_size, max_sentence_len)
  for k = 1, batch_size do
    sentence = sentences[start_index + k - 1]
    for i = 1, #sentence do 
      t[k][i] = sentence[i]
    end
    l[#l + 1] = t
  end
  batch_ru = t:clone()
  
  sentences = sentences_en
  t = torch.zeros(batch_size, max_sentence_len)
  for k = 1, batch_size do
    sentence = sentences[start_index + k - 1]
    for i = 1, #sentence do 
      t[k][i] = sentence[i]
    end
    l[#l + 1] = t
  end
  batch_en = t:clone()
  
  return batch_ru, batch_en
end

function gen_tensor_table(gen_ones)
  local h = {}
  for i = 1, opt.n_layers do 
    if gen_ones then
      h[#h + 1] = torch.ones(1, rnn_size):cuda()
    else  
      h[#h + 1] = torch.zeros(1, rnn_size):cuda()
    end
  end
  return h
end

lstm_c_enc0 = gen_tensor_table(false)
lstm_c_dec0 = gen_tensor_table(false)

-- do fwd/bwd and return loss, grad_params
function feval(x_arg)
    if x_arg ~= params then
        params:copy(x_arg)
    end
    grad_params:zero()
    
    ------------------- forward pass -------------------
    lstm_c_enc = {[0]=lstm_c_enc0}
    lstm_h_enc = {[0]=gen_tensor_table(false)}
    lstm_x_enc = {[0]=torch.zeros(1, rnn_size):cuda()}

    x_enc_embedding = {}
        
    local loss = 0
    
    x_enc = x_raw_enc[data_index]
    for t = 1, x_enc:size(2) - 1 do
      x_enc_embedding[t] = embed_enc_clones[t]:forward(x_enc[{{}, {t}}]:reshape(1):cuda())
      lstm_x_enc[t], lstm_c_enc[t], lstm_h_enc[t] = unpack(encoder_clones[t]:forward({x_enc_embedding[t], lstm_c_enc[t-1], lstm_h_enc[t-1]}))
    end
    
    lstm_c_dec = {[0]=lstm_c_dec0}
    lstm_h_dec = {[0]=lstm_h_enc[x_enc:size(2)-1]}
    lstm_c_enc0 = lstm_c_enc[x_enc:size(2)-1]
    x_dec_prediction = {}
    x_dec_embedding = {}
    
    y_dec = x_raw_dec[data_index]
    x_dec = torch.zeros(y_dec:size())
    x_dec[{{}, {1}}] = y_dec[{{}, {y_dec:size(2)}}]
    x_dec[{{}, {2,y_dec:size(2)}}] = y_dec[{{}, {1,y_dec:size(2) - 1}}]
    for t = 1, x_dec:size(2) do 
      x_dec_embedding[t] = embed_dec_clones[t]:forward(x_dec[{{}, {t}}]:reshape(1):cuda())
      lstm_c_dec[t], lstm_h_dec[t], x_dec_prediction[t] = unpack(decoder_clones[t]:forward({x_dec_embedding[t], lstm_c_dec[t-1], lstm_h_dec[t-1]}))
      loss_x = criterion_clones[t]:forward(x_dec_prediction[t], y_dec[{{}, {t}}]:reshape(1):cuda())
      loss = loss + loss_x
      --print(loss_x)
            
    end
    loss = loss / ((x_dec:size(2)) * n_layers)

    lstm_c_dec0 = lstm_c_dec[x_dec:size(2)]

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    dlstm_c_dec = {[x_dec:size(2)] = gen_tensor_table(false)}
    dlstm_h_dec = {[x_dec:size(2)] = gen_tensor_table(false)}
    dx_dec_prediction = {}
    dx_dec_embedding = {}
    dx_dec = {}
    dloss_x = {}
    
    for t = x_dec:size(2),1,-1 do
      dx_dec_prediction[t] = criterion_clones[t]:backward(x_dec_prediction[t], y_dec[{{}, {t}}]:reshape(1):cuda())
      dx_dec_embedding[t], dlstm_c_dec[t-1], dlstm_h_dec[t-1] = unpack(decoder_clones[t]:backward({x_dec_embedding[t], lstm_c_dec[t-1], lstm_h_dec[t-1]}, {dlstm_c_dec[t], dlstm_h_dec[t], dx_dec_prediction[t]}))
      dx_dec[t] = embed_dec_clones[t]:backward(x_dec[{{}, {t}}]:reshape(1):cuda(), dx_dec_embedding[t])
    end
    
    dlstm_c_enc = {[x_enc:size(2) - 1] = gen_tensor_table(false)}
    dlstm_h_enc = {[x_enc:size(2) - 1] = dlstm_h_dec[0]}
    dlstm_x_enc = {[x_enc:size(2) - 1] = torch.zeros(1, rnn_size):cuda()}
    dx_enc_embedding = {}
    dx_enc = {}

        
    for t = x_enc:size(2) -1, 1, -1 do
      dx_enc_embedding[t], dlstm_c_enc[t-1], dlstm_h_enc[t-1] = unpack(encoder_clones[t]:backward({x_enc_embedding[t], lstm_c_enc[t-1], lstm_h_enc[t-1]}, {dlstm_x_enc[t], dlstm_c_enc[t], dlstm_h_enc[t]}))
      dx_enc[{{}, {t}}] = embed_enc_clones[t]:backward(x_enc[{{}, {t}}]:reshape(1):cuda(), dx_enc_embedding[t])
    end
      
    -- clip gradient element-wise
    grad_params:clamp(-5, 5)
    data_index = data_index + 1
    if data_index > #x_raw_enc then 
      data_index = 1
    end

    return loss, grad_params
end

optim_state = {learningRate = 1e-2}


for i = 1, 2000000 do
  local _, loss = optim.adagrad(feval, params, optim_state)

  if i % 30 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      sample_sentence = {}
      target_sentence = {}
      source_sentence = {}
      for t = 1, x_dec:size(2) do 
        _, sampled_index = x_dec_prediction[t]:max(2)
        --print(sampled_index)
        sample_sentence[#sample_sentence + 1] = vocabulary_en[sampled_index[1][1]]
        target_sentence[#target_sentence + 1] = vocabulary_en[y_dec[1][t]]
     end
     for t = 1, x_enc:size(2) - 1 do 
        source_sentence[#source_sentence + 1] = vocabulary_ru[x_enc[1][t]]
     end
      print(table.concat(source_sentence, ' '))
      print(table.concat(sample_sentence, ' '))
      print(table.concat(target_sentence, ' '))
      
  end
end




