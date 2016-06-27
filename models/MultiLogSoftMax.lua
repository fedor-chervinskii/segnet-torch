------------------------------------------------------------------------
--[[ MultiLogSoftMax ]]--
-- Takes 2-3-4D input and performs a logsoftmax over a given dimension (default - the last).
-- Output is of the shape suitable for nn.ClassNLLCriterion
------------------------------------------------------------------------
local MultiLogSoftMax, parent = torch.class('MultiLogSoftMax', 'nn.Module')

function MultiLogSoftMax:__init(softmax_dimension)
   parent.__init(self)
   self.smdim = softmax_dimension or -1
   self.transposed_view = torch.LongStorage()
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function MultiLogSoftMax:updateOutput(input)
   self.input = input
   local Ndim = input:dim()
   if self.smdim == -1 then self.smdim = Ndim end
   if Ndim == 2 and self.smdim == 2 then
      return input.THNN.LogSoftMax_updateOutput(input:cdata(), self.output:cdata())
   end
   if Ndim > 4 or Ndim < 2 then
      error"Only supports 2-3-4D inputs"
   end
   local _input = input:transpose(self.smdim, Ndim):contiguous()
   local t_view = _input:size()
   local v_view = t_view[1]
   for i = 2,Ndim-1 do
        v_view = v_view*t_view[i]
   end
   self.transposed_view = t_view
   self._input = _input:view(v_view, t_view[Ndim])
   input.THNN.LogSoftMax_updateOutput(self._input:cdata(), self.output:cdata())
   return self.output
end

function MultiLogSoftMax:updateGradInput(input, gradOutput)
   local Ndim = input:dim()
   if input:dim() == 2 and self.smdim == 2 then
      return input.THNN.SoftMax_updateGradInput(input:cdata(), gradOutput:cdata(),
                                                self.gradInput:cdata(), self.output:cdata())
   end
   self.gradOutput = gradOutput
   input.THNN.SoftMax_updateGradInput(self._input:cdata(), gradOutput:cdata(),
                                      self.gradInput:cdata(), self.output:cdata())
   local gradInput = torch.view(self.gradInput, self.transposed_view)
   self.gradInput = gradInput:transpose(self.smdim, Ndim)
   return self.gradInput
end

return MultiLogSoftMax
