require 'image'
require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
require 'gnuplot'
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

IMFILE = '/media/ros/5CB4A980B4A95D70/lym/detection_model'
model = torch.load('./trained_models/mask/model.t7'):cuda()
model:evaluate()

-- View kernels
--[[
local d1 = image.toDisplayTensor{input=model:get(1).weight:clone():reshape(96, 3, 11, 11),
  padding = 2,
  nrow = math.floor(math.sqrt(96)),
  symmetric = true,
}
image.display{image=d1, legend='Layer 1 filters'}
-- Save h5 file
--local myFile = hdf5.open('./trained_models/cnnkernels.h5', 'w')
--myFile:write('/DS1', d1)
--myFile:close()
--]]

local imname = '00123.ppm'
local mean_std = torch.load('MEAN_STD.t7')
--local img_old = image.load(IMFILE..imname)

local img_old = image.load('./testimg1.jpg')
img_old = image.scale(img_old, 640, 480)

local img = img_old:clone()
normalize_global(img, mean_std[1], mean_std[2])
normalize_local(img)

local output = model:forward(img:cuda()):float()

-- Tiling
local outputnew = torch.Tensor(5, 120, 160):zero()
for idr = 1, 15 do
	for idc = 1, 20 do
	  local tmpr = { (idr-1)*8+1, idr*8 }
	  local tmpc = { (idc-1)*8+1, idc*8 }
		outputnew[{ {},tmpr,tmpc }] = torch.reshape(output[{ {},idr,idc }], 5, 8, 8)
	end
end
output = outputnew
local output = output:sum(1):squeeze()
output = output / output:max()
output = output:ge(0.6)

-- Generate bbox
local num = 10
local fH = output:size(1)
local fW = output:size(2)
local scale_h = torch.linspace(0.05, 0.3, num)
local scale_w = torch.linspace(0.05, 0.3, num)
local bbox_h = torch.round(scale_h * fH)
local bbox_w = torch.round(scale_w * fW)
bbox = torch.Tensor(num*num, 2):zero()
local ind = 1
for i = 1, num do
	for j = 1, num do
		bbox[ind][1] = bbox_h[i]
		bbox[ind][2] = bbox_w[j]
		ind = ind + 1
	end
end

-- Search bbox and get score
local bbox_all = {}
local bbox_score = {} 
for i = 1, fH, 3 do
	for j = 1, fW, 3 do

		for k = 1, bbox:size(1) do
			local top = i
			local left = j
			local bottom = math.min(fH, top+bbox[k][1]-1)
			local right = math.min(fW, left+bbox[k][2]-1)
			--local score = torch.Tensor(5):zero()
			-- Full
			local area_full = bbox[k][1] * bbox[k][2]
      local score = output[{ {top,bottom},{left,right} }]:sum() / area_full
			-- Halves
--[[
			local r_half = torch.floor(top+(bottom-top)/2)
			local c_half = torch.floor(left+(right-left)/2) 
			local area_top = (r_half-top+1)*(right-left+1)
			local area_bottom = area_full - area_top
			local area_left = (bottom-top+1)*(c_half-left+1)
			local area_right = area_full - area_left
			score[2] = output[{ 2,{top,r_half},{left,right} }]:sum() / area_top       --top
			score[3] = output[{ 3,{r_half,bottom},{left,right} }]:sum() / area_bottom  --bottom
			score[4] = output[{ 4,{top,bottom},{left,c_half} }]:sum() / area_left      --left
			score[5] = output[{ 5,{top,bottom},{c_half,right} }]:sum() / area_right    --right
--]]
			table.insert(bbox_all, {top, bottom, left, right})
			table.insert(bbox_score, score)
		end		
 
	end
end
bbox_score = torch.Tensor(bbox_score)
bbox_score = bbox_score / bbox_score:max()

-- Plot bbox
output = image.scale(output, 640, 480, 'simple')
local tmp = img_old[2]
tmp[output] = 1
img_old[2] = tmp

local th = 0.7
local vote = torch.Tensor(480, 640):zero()
for i = 1, (#bbox_score)[1] do
	if bbox_score[i] > th then
		local tmp = bbox_all[i]
		local top = tmp[1] * 4
		local bottom = tmp[2] * 4
		local left = tmp[3] * 4
		local right = tmp[4] * 4
		vote[{ {top,bottom},{left,right} }] = vote[{ {top,bottom},{left,right} }] + 1
		--draw.drawBox(img_old, top, bottom, left, right, 2, {math.random(), math.random(), math.random()})
		--d1 = image.display{image = img_old, win = d1}
	end
end
vote = vote:ge(3)

local location = vote:nonzero()
local len = location:size(1)
local vote_diff = location[{ {2,len},{} }] - location[{ {1,len-1},{} }]
local bwlabel = torch.Tensor(2, 480, 640):zero()
local indr, indc = 1, 1
for i = 2, len do
	-- Consider row
	if (vote_diff[i-1][1]==0 or vote_diff[i-1][1]==1) then
		bwlabel[{1,location[i][1],location[i][2]}] = indr
	else
		indr = indr + 1
		bwlabel[{1,location[i][1],location[i][2]}] = indr
	end
	-- Consider col
	if (vote_diff[i-1][2]==1) then	
		bwlabel[{2,location[i][1],location[i][2]}] = indc
	elseif (vote_diff[i-1][2]>1) then
		indc = indc + 1
		bwlabel[{2,location[i][1],location[i][2]}] = indc
	else
		indc = 1
		bwlabel[{2,location[i][1],location[i][2]}] = indc
	end
end
bwlabel = torch.cmul(bwlabel[1],bwlabel[1]) + bwlabel[2] - 1
local object_num = bwlabel:max()
gnuplot.imagesc(bwlabel)

for i = 1, object_num do
	local tmp = bwlabel:eq(i):nonzero()
	if (tmp:dim() > 0) then
		local top = tmp[{ {},1 }]:min()
		local bottom = tmp[{ {},1 }]:max()
		local left = tmp[{ {},2 }]:min()
		local right = tmp[{ {},2 }]:max()
		draw.drawBox(img_old, top, bottom, left, right, 2, {math.random(), math.random(), math.random()})
		d1 = image.display{image = img_old, win = d1}
	end
end























