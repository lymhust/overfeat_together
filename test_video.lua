require 'image'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'gnuplot'
require 'imgraph'
require 'ffmpeg'
require 'sys'
draw = require 'draw'
torch.setdefaulttensortype('torch.FloatTensor')

-- Functions
function normalize_global(dataset, mean, std)
  local std = std or dataset:std()
  local mean = mean or dataset:mean()
  dataset:add(-mean)
  dataset:div(std)
  return mean, std
end

function normalize_local(dataset)
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  normalized_images = norm:forward(dataset)
  dataset:copy(normalized_images)
end

function detection(frame)
	--local frame_proc = image.scale(frame, 640, 480)
	local frame_proc = frame:clone()
	normalize_global(frame_proc, mean_std[1], mean_std[2])
	normalize_local(frame_proc)
	local output = model:forward(frame_proc:cuda()):float()

	-- Tiling
	local outputnew = torch.Tensor(5, 120, 160):zero()
	for idr = 1, 15 do
	for idc = 1, 20 do
		local tmpr = { (idr-1)*8+1, idr*8 }
		local tmpc = { (idc-1)*8+1, idc*8 }
		outputnew[{ {},tmpr,tmpc }] = torch.reshape(output[{ {},idr,idc }], 5, 8, 8)
	end
	end
	output = image.scale(outputnew, 640, 480)

	-- Sum and thresh
	local th = 0.4
	local output = output:sum(1):squeeze() / 5
	output = output:ge(th)

	-- Merge and generate bbox
	-- bwlabel
	local graph = imgraph.graph(output:float())
	graph = imgraph.connectcomponents(graph, 0.1) + 10
	graph = torch.cmul(graph, output:float())
	local location = graph:nonzero()

	if (location:dim() > 0) then
		local val = graph[location[1][1]][location[1][2]]
		local indx = {}

		table.insert(indx, val)
		for i = 1, location:size(1) do
			local tmp = graph[location[i][1]][location[i][2]]
			local flag = true
			for j = 1, #indx do
				if (tmp == indx[j]) then
					flag = false
					break
				end
			end
			if (flag == true) then
				val = tmp
				table.insert(indx, val)
			end
		end

		-- Plot bbox
		local tmp = frame[2]
		tmp[output] = 1
		frame[2] = tmp
		local bias = 0
		for i = 1, #indx do
			local tmp = graph:eq(indx[i]):nonzero()
			if (tmp:dim() > 0) then
				local top = tmp[{ {},1 }]:min() - bias
				local bottom = tmp[{ {},1 }]:max() + bias
				local left = tmp[{ {},2 }]:min() - bias
				local right = tmp[{ {},2 }]:max() + bias
				if (bottom-top+1)>10 and (right-left+1)>10 then
					draw.drawBox(frame, top, bottom, left, right, 1, {math.random(), math.random(), math.random()})
				end
			end
		end
	end

	return frame
end
--------------------------------------------------

-- Load network
model = torch.load('./trained_models/model.t7'):cuda()
model:evaluate()
mean_std = torch.load('MEAN_STD.t7')

-- Load video
vid = ffmpeg.Video{path='camera.mp4', length=2000, width=640, height=480}

for i = 1, vid.nframes do
	sys.tic()
	
	local frame = vid:get_frame(1, i):float()
	frame = detection(frame)
	w = image.display{image = frame, win = w}

	print('FPS: '..1/sys.toc())
end
print('Finish')






