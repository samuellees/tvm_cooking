from tvm import relay
import tvm

x = relay.expr.var('x', relay.scalar_type('float32'), dtype = 'float32')
y = relay.expr.var('y', relay.scalar_type('float32'), dtype = 'float32')
dense = relay.nn.dense(x, y)    
func = relay.expr.Function([x], dense, relay.scalar_type('float32'))

mod = relay.Module.from_expr(func)  # note this API
print("Relay module function:\n", mod.astext(show_meta_data=False))
graph, lib, params = tvm.relay.build(mod, 'llvm', params={})
print("TVM graph:\n", graph)
print("TVM parameters:\n", params)
print("TVM compiled target function:\n", lib.get_source())