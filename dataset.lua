local M = {}
local FakeDataset = torch.class('resnet.FakeDataset', M)

function FakeDataset:__init(imageInfo, opt, split)
  -- imageInfo: result from dataset-gen.lua
  -- opt: command-line arguments
  -- split: "train" or "val"
end

function FakeDataset:get(i)
  return {
    input = torch.Tensor(3, 800, 600):uniform(),
    target = 42,
  }
end

function FakeDataset:size()
  -- size of dataset
  return 2000
end

function FakeDataset:preprocess()
  -- Scale smaller side to 256 and take 224x224 center-crop
  return t.Compose{
    t.Scale(256),
    t.CenterCrop(224),
  }
end

return M.FakeDataset
