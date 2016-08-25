require 'xlua'
require 'optim'
require 'pl'
require 'cudnn'
require 'cunn'
require 'gnuplot'
require 'image'
gnuplot.setterm('qt')
torch.setdefaulttensortype('torch.FloatTensor') 
local c = require 'trepl.colorize'

local opt = lapp[[
   -s,--save					(default "logs")                       subdirectory to save logs
   -b,--batchSize				(default 1)                            batch size
   -r,--learningRate            (default 1e-2)                         learning rate
   --learningRateDecay          (default 0)                            learning rate decay
   --weightDecay                (default 5e-4)                            weightDecay
   -m,--momentum                (default 0.9)                          momentum
   --epoch_step                 (default 1)                           epoch step
   --max_epoch                  (default 2)                          maximum number of iterations
   --type                       (default cuda)                         cuda or double
]]
print(opt)

-- Tiling functions
function tiling_backward(input, input_squeeze)
    -- Change groundtruth shape to output size
    for idr = 1, input_squeeze:size(3) do
        for idc = 1, input_squeeze:size(4) do
            local tmpr = { (idr-1)*8+1, idr*8 }
            local tmpc = { (idc-1)*8+1, idc*8 }
            input_squeeze[{ {},{},idr,idc }] = torch.reshape(input[{ {},{},tmpr,tmpc }], input_squeeze:size(2), 1)
        end
    end
end

function tiling_forward(input, input_exp)
    -- Tiling: change output to groundtruth shape
    for idr = 1, input:size(3) do
        for idc = 1, input:size(4) do
          local tmpr = { (idr-1)*8+1, idr*8 }
          local tmpc = { (idc-1)*8+1, idc*8 }
	      input_exp[{ {},{},tmpr,tmpc }] = torch.reshape(input[{ {},{},idr,idc }], input_exp:size(2), 8, 8)
        end
    end
end
--------------------------------------------------------


-- Setting parameters
local imH, imW = 128, 128
local outH, outW = imH/4, imW/4

-- Load data
local data = torch.load('./dataset/gtsdb_patch.t7')
IMG = data[1]
MASK = data[2]
BBOX = data[3]
--data = torch.load('./voc/voc12_trainData.t7')
--IMG = torch.cat(IMG, data[1], 1)
--MASK = torch.cat(MASK, data[3], 1)
print(#IMG)
print(#MASK)
print(#BBOX)

-- Log
print('Will save at '..opt.save)
paths.mkdir(opt.save)
local testLogger = optim.Logger(paths.concat(opt.save, 'train_info.log'))
testLogger:setNames{'% cost (train set)'}
testLogger.showPlot = false

-- Load model
print(c.red('==> load model'))
--share, branch1, branch2 = dofile('generate_model.lua')
mod = torch.load('./trained_models/model_two_branch_iter_12.t7')
--share = torch.load('./trained_models/share_1.t7'):cuda()
--branch1 = torch.load('./trained_models/branch1_1.t7'):cuda()
--branch2 = torch.load('./trained_models/branch2_1.t7'):cuda()
share = mod[1]:cuda()
branch1 = mod[2]:cuda() -- Mask
branch2 = mod[3]:cuda() -- BBOX
-- Concat
local mlp = nn.ConcatTable()
mlp:add(branch1)
mlp:add(branch2)
-- Add branches
model = nn.Sequential()
model:add(share)
model:add(mlp)
model = model:cuda()
print(model)

parameters, gradParameters = model:getParameters()

-- Set criterion
print(c.red('==> setting criterion'))
criterion_b1 = cudnn.SpatialCrossEntropyCriterion():cuda() 
criterion_b2 = nn.MSECriterion():cuda() 

-- Set optimizer
print(c.red('==> configuring optimizer'))
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

inputs = torch.CudaTensor(opt.batchSize, 3, imH, imW):zero()
target_b1 = torch.CudaTensor(opt.batchSize, outH, outW):zero()    -- Mask
target_b2 = torch.CudaTensor(opt.batchSize, 4, outH, outW):zero() -- BBox
output_b1 = torch.CudaTensor(opt.batchSize, 2, outH, outW):zero()
output_b2 = torch.CudaTensor(opt.batchSize, 4, outH, outW):zero()
df_c1_reshape = torch.CudaTensor(opt.batchSize, 2*64, outH/8, outW/8):zero()
df_c2_reshape = torch.CudaTensor(opt.batchSize, 4*64, outH/8, outW/8):zero()

function train()
    
  -- Training mod
  share:training()
  branch1:training()
  branch2:training()

  local cost = {}
  
  
  --if opt.type == 'double' then inputs = inputs:double()
  --elseif opt.type == 'cuda' then inputs = inputs:cuda() target_b1 = target_b1:cuda() target_b2 = target_b2:cuda() end

  --epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  -- shuffle at each epoch
  local trainnum = IMG:size(1)
  shuffle = torch.randperm(trainnum)

  for t = 1, trainnum, opt.batchSize do
    -- Disp progress
    xlua.progress(t, trainnum)

    -- Create mini batch
    local indx = 1
    inputs:zero()
    target_b1:zero()
    target_b2:zero()
    output_b1:zero()
    output_b2:zero()
    df_c1_reshape:zero()
    df_c2_reshape:zero()
    for i = t, math.min(t+opt.batchSize-1, trainnum) do
      -- Load new sample
      inputs[indx] = IMG[shuffle[i]]
      target_b1[indx] = MASK[shuffle[i]]
      target_b2[indx] = BBOX[shuffle[i]]
      indx = indx + 1
    end
    
    -- Create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      
      -- reset gradients
      gradParameters:zero()
      
      -- evaluate function for complete mini batch
      -- estimate f
      local output_sh = share:forward(inputs)
      tiling_forward(branch1:forward(output_sh), output_b1)
      tiling_forward(branch2:forward(output_sh), output_b2)
      
      local f_b1 = criterion_b1:forward(output_b1, target_b1)
      local f_b2 = criterion_b2:forward(output_b2, target_b2)
      local f = f_b1 + f_b2
      
      -- estimate df/dW
      local df_c1 = criterion_b1:backward(output_b1, target_b1)
      local df_c2 = criterion_b2:backward(output_b2, target_b2)
      tiling_backward(df_c1, df_c1_reshape)
      tiling_backward(df_c2, df_c2_reshape)
      
      local df_b1 = branch1:backward(output_sh, df_c1_reshape)
      local df_b2 = branch2:backward(output_sh, df_c2_reshape)
      
      local df_sh = df_b1 + df_b2
      share:backward(inputs, df_sh)
      
      -- normalize gradients and f(X)
      gradParameters:div(inputs:size(1))
      
      -- Visualize
      ---[[
      local dd1 = image.toDisplayTensor{input=output_b1[{ 1,{2},{},{} }]:squeeze():gt(0),
                                        padding = 2,
                                        nrow = math.floor(math.sqrt(64)),
                                        symmetric = true}
      d1 = image.display{image = dd1, win = d1}
      local dd2 = image.toDisplayTensor{input=target_b1[{ 1,{},{} }]:squeeze(),
                                        padding = 2,
                                        nrow = math.floor(math.sqrt(64)),
                                        symmetric = true}
      d2 = image.display{image = dd2, win = d2}
      local dd3 = image.toDisplayTensor{input=share:get(1).weight:clone():reshape(96, 3, 11, 11),
                                        padding = 2,
                                        nrow = math.floor(math.sqrt(96)),
                                        symmetric = true}
      d3 = image.display{image=dd3, win=d3, legend='Layer 1 filters'}
      --]]
      
      table.insert(cost, f)
      
      -- return f and df/dX
      return f, gradParameters
    end -- Local function

    optim.sgd(feval, parameters, optimState)

    --gnuplot.plot(torch.Tensor(cost))

  end -- trainnum
  print("  Mean cost: "..torch.Tensor(cost):mean())
  testLogger:add{"  Mean cost: "..torch.Tensor(cost):mean()}
end -- Train function


for i = 1,  opt.max_epoch do

  print('Epoch '..i)
  testLogger:add{'Epoch '..i}
  train()
  
  if (math.fmod(i, opt.epoch_step) == 0) then
    -- Save model
    torch.save('./trained_models/model_'..i..'.t7', {share,branch1,branch2})
  end
  
end




        




