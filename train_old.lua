require 'xlua'
require 'optim'
require 'pl'
require 'cudnn'
require 'cunn'
require 'gnuplot'
require 'image'
local c = require 'trepl.colorize'


opt = lapp[[
   -s,--save					(default "logs")                       subdirectory to save logs
   -b,--batchSize				(default 1)                            batch size
   -r,--learningRate            (default 1e-2)                         learning rate
   --learningRateDecay          (default 0)                         learning rate decay
   --weightDecay                (default 0)                         weightDecay
   -m,--momentum                (default 0.9)                          momentum
   --epoch_step                 (default 25)                           epoch step
   --max_epoch                  (default 100)                          maximum number of iterations
   --type                       (default cuda)                         cuda or double
]]

print(opt)

DATA_POS = torch.load('dataset/gtsdb_patch.t7')
IMGPATCH = DATA_POS[1]
IMGMASK = DATA_POS[2]

print(#IMGPATCH)
print(#IMGMASK)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

print(c.red'==>'..c.red' load model')
model = dofile('generate_model_1.lua'):cuda()
--model = torch.load('./trained_models/model.t7'):cuda()
parameters, gradParameters = model:getParameters()
print(model)

print(c.red'==>' ..c.red' setting criterion')
criterion = nn.MSECriterion():cuda()

print(c.red'==>'..c.red' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

function train()

  model:training()

  local cost = {}
  local inputs = torch.Tensor(opt.batchSize, 3, 128, 128):zero()
  local targets = torch.Tensor(opt.batchSize, 64*5, 4, 4):zero()
  if opt.type == 'double' then inputs = inputs:double()
  elseif opt.type == 'cuda' then inputs = inputs:cuda() targets = targets:cuda() end

  --epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  -- shuffle at each epoch
  local trainnum = IMGPATCH:size(1)
  shuffle = torch.randperm(trainnum)

  for t = 1, trainnum, opt.batchSize do
    -- Disp progress
    xlua.progress(t, trainnum)

    -- Create mini batch
    local indx = 1
    targets:zero()
    for i = t, math.min(t+opt.batchSize-1, trainnum) do
      -- Load new sample
      inputs[indx] = IMGPATCH[shuffle[i]]
      local mask = IMGMASK[shuffle[i]]
      -- Change mask shape
      for idr = 1, 4 do
      	for idc = 1, 4 do
      	  local tmpr = { (idr-1)*8+1, idr*8 }
      	  local tmpc = { (idc-1)*8+1, idc*8 }
      		targets[{ indx,{},idr,idc }] = torch.reshape(mask[{ {},tmpr,tmpc }], 64*5, 1)
      	end
      end
      indx = indx + 1
    end
		
    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end
      -- reset gradients
      gradParameters:zero()
      -- evaluate function for complete mini batch
      -- estimate f
      local output = model:forward(inputs)
      
      -- Tiling
      ---[[
      local outputnew = torch.CudaTensor(1, 5, 32, 32):zero()
      local targetnew = torch.CudaTensor(1, 5, 32, 32):zero()
      for idr = 1, 4 do
      	for idc = 1, 4 do
      	  local tmpr = { (idr-1)*8+1, idr*8 }
      	  local tmpc = { (idc-1)*8+1, idc*8 }
      		outputnew[{ 1,{},tmpr,tmpc }] = torch.reshape(output[{ 1,{},idr,idc }], 5, 8, 8)
      		targetnew[{ 1,{},tmpr,tmpc }] = torch.reshape(targets[{ 1,{},idr,idc }], 5, 8, 8)
      	end
      end
      --]]
      
      ---[[
			local dd1 = image.toDisplayTensor{input=outputnew[{ 1,{},{},{} }]:squeeze(),
				padding = 2,
				nrow = math.floor(math.sqrt(64)),
				symmetric = true,
			}
      d1 = image.display{image = dd1, win = d1}
			local dd2 = image.toDisplayTensor{input=targetnew[{ 1,{},{},{} }]:squeeze(),
				padding = 2,
				nrow = math.floor(math.sqrt(64)),
				symmetric = true,
			}
			d2 = image.display{image = dd2, win = d2}
			local dd3 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
				padding = 2,
				nrow = math.floor(math.sqrt(96)),
				symmetric = true,
			}
			d3 = image.display{image=dd3, win=d3, legend='Layer 1 filters'}
			--]]
      
      -- local weights = targets:eq(1) * 10  
      --criterion = nn.BCECriterion(weights:float():view(-1)):cuda()
        
      local f = criterion:forward(output, targets)
      table.insert(cost, f)
      -- estimate df/dW
      local df_do = criterion:backward(output, targets)
      model:backward(inputs, df_do)
      -- normalize gradients and f(X)
      gradParameters:div(inputs:size(1))
      -- return f and df/dX
      return f, gradParameters
    end

    optim.sgd(feval, parameters, optimState)

    --gnuplot.plot(torch.Tensor(cost))

  end
  print("  Mean cost: "..torch.Tensor(cost):mean())
end

--[[
function test()

  model:evaluate()

  -- Test on train, test and validate images
  local function testfunc_train(NameList, Labels)
    local batchsize = 20
    local sizeall = batchsize - math.fmod(#NameList, batchsize) + #NameList
    local inputs = torch.Tensor(batchsize, 3, 224, 224)
    local targets = torch.Tensor(sizeall)
    local outputs = torch.Tensor(sizeall)
    local indxall = 1;
    if opt.type == 'double' then inputs = inputs:double()
    elseif opt.type == 'cuda' then inputs = inputs:cuda()  end

    for t = 1, sizeall, batchsize do
      xlua.progress(t, sizeall)
      local indx = 1
      for i = t, math.min(t+batchsize-1, sizeall) do
        -- load new sample
        local tpname = NameList[i]
        if (tpname ~= nil) then
          tpname = './datas/LIVE_imagepatches/'..tpname
          inputs[{ indx, {},{},{} }]= preprocess(image.load(tpname), img_mean)
          targets[indxall] = Labels[i]
        else
          inputs[{ indx,{},{},{} }]= inputs[{ 1,{},{},{} }]
          targets[indxall] = Labels[1]
        end
        indx = indx + 1
        indxall  = indxall + 1
      end
      outputs[{ {t,t+batchsize-1} }] = model:forward(inputs):double():squeeze()
    end
    targets = targets[{ {1,#NameList} }]
    outputs = outputs[{ {1,#NameList} }]
    local lcc = lcc(targets, outputs)
    local srocc = srocc(targets, outputs)
    return lcc, srocc, targets, outputs
  end

  local function testfunc_test(NameList, Labels)
    local batchsize = 1
    local sizeall = batchsize - math.fmod(#NameList, batchsize) + #NameList
    local inputs = torch.Tensor(batchsize, 3, 448, 448)
    local targets = torch.Tensor(sizeall)
    local outputs = torch.Tensor(sizeall)
    local indxall = 1;
    if opt.type == 'double' then inputs = inputs:double()
    elseif opt.type == 'cuda' then inputs = inputs:cuda()  end

    for t = 1, sizeall, batchsize do
      xlua.progress(t, sizeall)
      local indx = 1
      for i = t, math.min(t+batchsize-1, sizeall) do
        -- load new sample
        local tpname = NameList[i]
        if (tpname ~= nil) then
          tpname = './datas/LIVE_images/'..tpname
          inputs[{ indx, {},{},{} }]= preprocess_large(image.load(tpname), img_mean_large)
          targets[indxall] = Labels[i]
        else
          inputs[{ indx,{},{},{} }]= inputs[{ 1,{},{},{} }]
          targets[indxall] = Labels[1]
        end
        indx = indx + 1
        indxall  = indxall + 1
      end
      outputs[{ {t,t+batchsize-1} }] = model:forward(inputs):double():squeeze():mean()
    end
    targets = targets[{ {1,#NameList} }]
    outputs = outputs[{ {1,#NameList} }]
    local lcc = lcc(targets, outputs)
    local srocc = srocc(targets, outputs)
    return lcc, srocc, targets, outputs
  end

  local trainlcc, trainsrocc = testfunc_train(trainNameList, trainLabels)
  local testlcc, testsrocc, targets, outputs = testfunc_test(testNameList, testLabels)

  print('LCC train: '..trainlcc..' SROCC train: '..trainsrocc)
  print('LCC test: '..testlcc..' SROCC test: '..testsrocc)

  return trainlcc, trainsrocc, testlcc, testsrocc, targets, outputs
end
--]]

for i = 1,  opt.max_epoch do
  print('Epoch '..i)
  train()
  
  -- View kernels
	--[[
	local d1 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
		padding = 2,
		nrow = math.floor(math.sqrt(96)),
		symmetric = true,
	}
	image.display{image=d1, legend='Layer 1 filters'}
	--]]
	
  --if (i  == 1 or math.fmod(i, 10) == 0) then
    --print('Epoch '..i)
    --trainlcc, trainsrocc, testlcc, testsrocc, testtarg, testout = test()
  --end
end

-- Save model
torch.save('./trained_models/model.t7', model)


