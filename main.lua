--
--  Copyright (c) 2016, Facebook, Inc.
--  Copyright (c) 2016, Fedor Chervinskii
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'

local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local cacc, acc = trainer:test(0, valLoader)
   print(string.format(' * Results avg.class acc: %6.3f  avg.acc: %6.3f', cacc, acc))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local cacc = 0
local acc = 0
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      print(' * Best model ', testTop1, testTop5)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel)
end

print(string.format(' * Results avg.class acc: %6.3f  avg.acc: %6.3f', cacc, acc))
