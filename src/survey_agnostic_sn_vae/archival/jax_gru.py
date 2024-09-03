from jax import numpy as jnp
from jax import jit
import haiku as hk


def GRUHaiku(out_dim):
    class CustomGRUModule(hk.Module):
        def __init__(self):
            super().__init__()
            self.gru = hk.GRU(out_dim)

            @jit
            def step_fn(prev_state, x):
                out = self.gru(x, prev_state)
                return out[1], out[0]
                    
            @jit
            def for_loop(state, inputs):
                
                for i in inputs:
                    out, state = self.step(state, i)
                    
                return out

            self.step = step_fn
            self.forward = for_loop
            
            
        def __call__(self, inputs):
            inputs = jnp.transpose(inputs, (1, 0, 2)) # time needs to be first dimension
            seq_length, batch_size, input_dim = inputs.shape
    
            # Initialize state
            initial_state = self.gru.initial_state(batch_size)

            self.forward(initial_state, inputs)
            # Apply GRU over sequence dimension
            #state, outputs = hk.scan(self.step, initial_state, inputs)
            #return outputs[-1]
            
            

    def my_haiku_module_fn(inputs, training):
        module = CustomGRUModule()
        return module(inputs)
        
    transformed_module =  hk.transform(my_haiku_module_fn)
    init_fn = transformed_module.init
    call_fn = transformed_module.apply

    return {
        'call_fn': call_fn,
        'init_fn': init_fn,
    }
