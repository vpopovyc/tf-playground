
P
inputs/inputPlaceholder*&
shape:’’’’’’’’’¬į*
dtype0
P
inputs/correct_answersPlaceholder*
shape:’’’’’’’’’*
dtype0
[
weights/truncated_normal/shapeConst*
dtype0*%
valueB"            
J
weights/truncated_normal/meanConst*
valueB
 *    *
dtype0
L
weights/truncated_normal/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

(weights/truncated_normal/TruncatedNormalTruncatedNormalweights/truncated_normal/shape*
seed2*

seed *
T0*
dtype0
w
weights/truncated_normal/mulMul(weights/truncated_normal/TruncatedNormalweights/truncated_normal/stddev*
T0
e
weights/truncated_normalAddweights/truncated_normal/mulweights/truncated_normal/mean*
T0
l
weights/weight_1
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¤
weights/weight_1/AssignAssignweights/weight_1weights/truncated_normal*#
_class
loc:@weights/weight_1*
validate_shape(*
use_locking(*
T0
a
weights/weight_1/readIdentityweights/weight_1*#
_class
loc:@weights/weight_1*
T0
]
 weights/truncated_normal_1/shapeConst*%
valueB"            *
dtype0
L
weights/truncated_normal_1/meanConst*
valueB
 *    *
dtype0
N
!weights/truncated_normal_1/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

*weights/truncated_normal_1/TruncatedNormalTruncatedNormal weights/truncated_normal_1/shape*
T0*
dtype0*
seed2*

seed 
}
weights/truncated_normal_1/mulMul*weights/truncated_normal_1/TruncatedNormal!weights/truncated_normal_1/stddev*
T0
k
weights/truncated_normal_1Addweights/truncated_normal_1/mulweights/truncated_normal_1/mean*
T0
l
weights/weight_2
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¦
weights/weight_2/AssignAssignweights/weight_2weights/truncated_normal_1*
use_locking(*
T0*#
_class
loc:@weights/weight_2*
validate_shape(
a
weights/weight_2/readIdentityweights/weight_2*
T0*#
_class
loc:@weights/weight_2
]
 weights/truncated_normal_2/shapeConst*%
valueB"            *
dtype0
L
weights/truncated_normal_2/meanConst*
valueB
 *    *
dtype0
N
!weights/truncated_normal_2/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

*weights/truncated_normal_2/TruncatedNormalTruncatedNormal weights/truncated_normal_2/shape*
T0*
dtype0*
seed2*

seed 
}
weights/truncated_normal_2/mulMul*weights/truncated_normal_2/TruncatedNormal!weights/truncated_normal_2/stddev*
T0
k
weights/truncated_normal_2Addweights/truncated_normal_2/mulweights/truncated_normal_2/mean*
T0
l
weights/weight_3
VariableV2*
dtype0*
	container *
shape:*
shared_name 
¦
weights/weight_3/AssignAssignweights/weight_3weights/truncated_normal_2*
use_locking(*
T0*#
_class
loc:@weights/weight_3*
validate_shape(
a
weights/weight_3/readIdentityweights/weight_3*#
_class
loc:@weights/weight_3*
T0
U
 weights/truncated_normal_3/shapeConst*
valueB":  Č   *
dtype0
L
weights/truncated_normal_3/meanConst*
valueB
 *    *
dtype0
N
!weights/truncated_normal_3/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

*weights/truncated_normal_3/TruncatedNormalTruncatedNormal weights/truncated_normal_3/shape*
dtype0*
seed2 *

seed *
T0
}
weights/truncated_normal_3/mulMul*weights/truncated_normal_3/TruncatedNormal!weights/truncated_normal_3/stddev*
T0
k
weights/truncated_normal_3Addweights/truncated_normal_3/mulweights/truncated_normal_3/mean*
T0
f
weights/weight_4
VariableV2*
shared_name *
dtype0*
	container *
shape:
uČ
¦
weights/weight_4/AssignAssignweights/weight_4weights/truncated_normal_3*
validate_shape(*
use_locking(*
T0*#
_class
loc:@weights/weight_4
a
weights/weight_4/readIdentityweights/weight_4*
T0*#
_class
loc:@weights/weight_4
U
 weights/truncated_normal_4/shapeConst*
valueB"Č      *
dtype0
L
weights/truncated_normal_4/meanConst*
valueB
 *    *
dtype0
N
!weights/truncated_normal_4/stddevConst*
valueB
 *ĶĢĢ=*
dtype0

*weights/truncated_normal_4/TruncatedNormalTruncatedNormal weights/truncated_normal_4/shape*
T0*
dtype0*
seed2)*

seed 
}
weights/truncated_normal_4/mulMul*weights/truncated_normal_4/TruncatedNormal!weights/truncated_normal_4/stddev*
T0
k
weights/truncated_normal_4Addweights/truncated_normal_4/mulweights/truncated_normal_4/mean*
T0
e
weights/weight_5
VariableV2*
shared_name *
dtype0*
	container *
shape:	Č
¦
weights/weight_5/AssignAssignweights/weight_5weights/truncated_normal_4*
validate_shape(*
use_locking(*
T0*#
_class
loc:@weights/weight_5
a
weights/weight_5/readIdentityweights/weight_5*
T0*#
_class
loc:@weights/weight_5
I
biases/ones/shape_as_tensorConst*
valueB:*
dtype0
>
biases/ones/ConstConst*
valueB
 *  ?*
dtype0
^
biases/onesFillbiases/ones/shape_as_tensorbiases/ones/Const*
T0*

index_type0
=
biases/truediv/yConst*
valueB
 *  @*
dtype0
A
biases/truedivRealDivbiases/onesbiases/truediv/y*
T0
]
biases/bias_1
VariableV2*
dtype0*
	container *
shape:*
shared_name 

biases/bias_1/AssignAssignbiases/bias_1biases/truediv*
validate_shape(*
use_locking(*
T0* 
_class
loc:@biases/bias_1
X
biases/bias_1/readIdentitybiases/bias_1*
T0* 
_class
loc:@biases/bias_1
K
biases/ones_1/shape_as_tensorConst*
valueB:*
dtype0
@
biases/ones_1/ConstConst*
valueB
 *  ?*
dtype0
d
biases/ones_1Fillbiases/ones_1/shape_as_tensorbiases/ones_1/Const*
T0*

index_type0
?
biases/truediv_1/yConst*
valueB
 *  @*
dtype0
G
biases/truediv_1RealDivbiases/ones_1biases/truediv_1/y*
T0
]
biases/bias_2
VariableV2*
shared_name *
dtype0*
	container *
shape:

biases/bias_2/AssignAssignbiases/bias_2biases/truediv_1* 
_class
loc:@biases/bias_2*
validate_shape(*
use_locking(*
T0
X
biases/bias_2/readIdentitybiases/bias_2* 
_class
loc:@biases/bias_2*
T0
K
biases/ones_2/shape_as_tensorConst*
valueB:*
dtype0
@
biases/ones_2/ConstConst*
valueB
 *  ?*
dtype0
d
biases/ones_2Fillbiases/ones_2/shape_as_tensorbiases/ones_2/Const*
T0*

index_type0
?
biases/truediv_2/yConst*
dtype0*
valueB
 *  @
G
biases/truediv_2RealDivbiases/ones_2biases/truediv_2/y*
T0
]
biases/bias_3
VariableV2*
dtype0*
	container *
shape:*
shared_name 

biases/bias_3/AssignAssignbiases/bias_3biases/truediv_2*
use_locking(*
T0* 
_class
loc:@biases/bias_3*
validate_shape(
X
biases/bias_3/readIdentitybiases/bias_3*
T0* 
_class
loc:@biases/bias_3
L
biases/ones_3/shape_as_tensorConst*
valueB:Č*
dtype0
@
biases/ones_3/ConstConst*
dtype0*
valueB
 *  ?
d
biases/ones_3Fillbiases/ones_3/shape_as_tensorbiases/ones_3/Const*

index_type0*
T0
?
biases/truediv_3/yConst*
valueB
 *  @*
dtype0
G
biases/truediv_3RealDivbiases/ones_3biases/truediv_3/y*
T0
^
biases/bias_4
VariableV2*
	container *
shape:Č*
shared_name *
dtype0

biases/bias_4/AssignAssignbiases/bias_4biases/truediv_3*
validate_shape(*
use_locking(*
T0* 
_class
loc:@biases/bias_4
X
biases/bias_4/readIdentitybiases/bias_4* 
_class
loc:@biases/bias_4*
T0
K
biases/ones_4/shape_as_tensorConst*
valueB:*
dtype0
@
biases/ones_4/ConstConst*
dtype0*
valueB
 *  ?
d
biases/ones_4Fillbiases/ones_4/shape_as_tensorbiases/ones_4/Const*
T0*

index_type0
?
biases/truediv_4/yConst*
valueB
 *  @*
dtype0
G
biases/truediv_4RealDivbiases/ones_4biases/truediv_4/y*
T0
]
biases/bias_5
VariableV2*
shape:*
shared_name *
dtype0*
	container 

biases/bias_5/AssignAssignbiases/bias_5biases/truediv_4*
use_locking(*
T0* 
_class
loc:@biases/bias_5*
validate_shape(
X
biases/bias_5/readIdentitybiases/bias_5*
T0* 
_class
loc:@biases/bias_5
²
model/Conv2DConv2Dinputs/inputweights/weight_1/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
;
	model/addAddmodel/Conv2Dbiases/bias_1/read*
T0
$
model/L1Relu	model/add*
T0
°
model/Conv2D_1Conv2Dmodel/L1weights/weight_2/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
model/add_1Addmodel/Conv2D_1biases/bias_2/read*
T0
&
model/L2Relumodel/add_1*
T0
°
model/Conv2D_2Conv2Dmodel/L2weights/weight_3/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
model/add_2Addmodel/Conv2D_2biases/bias_3/read*
T0
&
model/L3Relumodel/add_2*
T0
C
model/LF/shapeConst*
valueB"’’’’:  *
dtype0
D
model/LFReshapemodel/L3model/LF/shape*
T0*
Tshape0
f
model/MatMulMatMulmodel/LFweights/weight_4/read*
transpose_a( *
transpose_b( *
T0
=
model/add_3Addmodel/MatMulbiases/bias_4/read*
T0
k
model/MatMul_1MatMulmodel/add_3weights/weight_5/read*
T0*
transpose_a( *
transpose_b( 
?
model/add_4Addmodel/MatMul_1biases/bias_5/read*
T0
.
model/SoftmaxSoftmaxmodel/add_4*
T0
e
4evalution_metrics/cross_entropy/labels_stop_gradientStopGradientinputs/correct_answers*
T0
N
$evalution_metrics/cross_entropy/RankConst*
value	B :*
dtype0
V
%evalution_metrics/cross_entropy/ShapeShapemodel/Softmax*
out_type0*
T0
P
&evalution_metrics/cross_entropy/Rank_1Const*
value	B :*
dtype0
X
'evalution_metrics/cross_entropy/Shape_1Shapemodel/Softmax*
T0*
out_type0
O
%evalution_metrics/cross_entropy/Sub/yConst*
value	B :*
dtype0

#evalution_metrics/cross_entropy/SubSub&evalution_metrics/cross_entropy/Rank_1%evalution_metrics/cross_entropy/Sub/y*
T0
v
+evalution_metrics/cross_entropy/Slice/beginPack#evalution_metrics/cross_entropy/Sub*
T0*

axis *
N
X
*evalution_metrics/cross_entropy/Slice/sizeConst*
valueB:*
dtype0
Ę
%evalution_metrics/cross_entropy/SliceSlice'evalution_metrics/cross_entropy/Shape_1+evalution_metrics/cross_entropy/Slice/begin*evalution_metrics/cross_entropy/Slice/size*
T0*
Index0
f
/evalution_metrics/cross_entropy/concat/values_0Const*
valueB:
’’’’’’’’’*
dtype0
U
+evalution_metrics/cross_entropy/concat/axisConst*
value	B : *
dtype0
Õ
&evalution_metrics/cross_entropy/concatConcatV2/evalution_metrics/cross_entropy/concat/values_0%evalution_metrics/cross_entropy/Slice+evalution_metrics/cross_entropy/concat/axis*
N*

Tidx0*
T0

'evalution_metrics/cross_entropy/ReshapeReshapemodel/Softmax&evalution_metrics/cross_entropy/concat*
T0*
Tshape0
P
&evalution_metrics/cross_entropy/Rank_2Const*
dtype0*
value	B :

'evalution_metrics/cross_entropy/Shape_2Shape4evalution_metrics/cross_entropy/labels_stop_gradient*
out_type0*
T0
Q
'evalution_metrics/cross_entropy/Sub_1/yConst*
value	B :*
dtype0

%evalution_metrics/cross_entropy/Sub_1Sub&evalution_metrics/cross_entropy/Rank_2'evalution_metrics/cross_entropy/Sub_1/y*
T0
z
-evalution_metrics/cross_entropy/Slice_1/beginPack%evalution_metrics/cross_entropy/Sub_1*
T0*

axis *
N
Z
,evalution_metrics/cross_entropy/Slice_1/sizeConst*
valueB:*
dtype0
Ģ
'evalution_metrics/cross_entropy/Slice_1Slice'evalution_metrics/cross_entropy/Shape_2-evalution_metrics/cross_entropy/Slice_1/begin,evalution_metrics/cross_entropy/Slice_1/size*
T0*
Index0
h
1evalution_metrics/cross_entropy/concat_1/values_0Const*
valueB:
’’’’’’’’’*
dtype0
W
-evalution_metrics/cross_entropy/concat_1/axisConst*
dtype0*
value	B : 
Ż
(evalution_metrics/cross_entropy/concat_1ConcatV21evalution_metrics/cross_entropy/concat_1/values_0'evalution_metrics/cross_entropy/Slice_1-evalution_metrics/cross_entropy/concat_1/axis*
T0*
N*

Tidx0
«
)evalution_metrics/cross_entropy/Reshape_1Reshape4evalution_metrics/cross_entropy/labels_stop_gradient(evalution_metrics/cross_entropy/concat_1*
Tshape0*
T0

evalution_metrics/cross_entropySoftmaxCrossEntropyWithLogits'evalution_metrics/cross_entropy/Reshape)evalution_metrics/cross_entropy/Reshape_1*
T0
Q
'evalution_metrics/cross_entropy/Sub_2/yConst*
dtype0*
value	B :

%evalution_metrics/cross_entropy/Sub_2Sub$evalution_metrics/cross_entropy/Rank'evalution_metrics/cross_entropy/Sub_2/y*
T0
[
-evalution_metrics/cross_entropy/Slice_2/beginConst*
valueB: *
dtype0
y
,evalution_metrics/cross_entropy/Slice_2/sizePack%evalution_metrics/cross_entropy/Sub_2*
T0*

axis *
N
Ź
'evalution_metrics/cross_entropy/Slice_2Slice%evalution_metrics/cross_entropy/Shape-evalution_metrics/cross_entropy/Slice_2/begin,evalution_metrics/cross_entropy/Slice_2/size*
T0*
Index0

)evalution_metrics/cross_entropy/Reshape_2Reshapeevalution_metrics/cross_entropy'evalution_metrics/cross_entropy/Slice_2*
T0*
Tshape0
E
evalution_metrics/ConstConst*
dtype0*
valueB: 

evalution_metrics/MeanMean)evalution_metrics/cross_entropy/Reshape_2evalution_metrics/Const*
T0*
	keep_dims( *

Tidx0
L
"evalution_metrics/ArgMax/dimensionConst*
value	B :*
dtype0
}
evalution_metrics/ArgMaxArgMaxmodel/Softmax"evalution_metrics/ArgMax/dimension*
T0*
output_type0	*

Tidx0
N
$evalution_metrics/ArgMax_1/dimensionConst*
value	B :*
dtype0

evalution_metrics/ArgMax_1ArgMaxinputs/correct_answers$evalution_metrics/ArgMax_1/dimension*
T0*
output_type0	*

Tidx0
_
evalution_metrics/EqualEqualevalution_metrics/ArgMaxevalution_metrics/ArgMax_1*
T0	
O
evalution_metrics/CastCastevalution_metrics/Equal*

SrcT0
*

DstT0
G
evalution_metrics/Const_1Const*
valueB: *
dtype0
y
evalution_metrics/Mean_1Meanevalution_metrics/Castevalution_metrics/Const_1*
	keep_dims( *

Tidx0*
T0
J
 optimizer/Variable/initial_valueConst*
value	B : *
dtype0
^
optimizer/Variable
VariableV2*
shape: *
shared_name *
dtype0*
	container 
²
optimizer/Variable/AssignAssignoptimizer/Variable optimizer/Variable/initial_value*
use_locking(*
T0*%
_class
loc:@optimizer/Variable*
validate_shape(
g
optimizer/Variable/readIdentityoptimizer/Variable*
T0*%
_class
loc:@optimizer/Variable
U
(optimizer/ExponentialDecay/learning_rateConst*
valueB
 *RI9*
dtype0
X
optimizer/ExponentialDecay/CastCastoptimizer/Variable/read*

SrcT0*

DstT0
O
#optimizer/ExponentialDecay/Cast_1/xConst*
dtype0*
valueB	 :Č
f
!optimizer/ExponentialDecay/Cast_1Cast#optimizer/ExponentialDecay/Cast_1/x*

SrcT0*

DstT0
P
#optimizer/ExponentialDecay/Cast_2/xConst*
valueB
 *Āu?*
dtype0
z
"optimizer/ExponentialDecay/truedivRealDivoptimizer/ExponentialDecay/Cast!optimizer/ExponentialDecay/Cast_1*
T0
V
 optimizer/ExponentialDecay/FloorFloor"optimizer/ExponentialDecay/truediv*
T0
u
optimizer/ExponentialDecay/PowPow#optimizer/ExponentialDecay/Cast_2/x optimizer/ExponentialDecay/Floor*
T0
t
optimizer/ExponentialDecayMul(optimizer/ExponentialDecay/learning_rateoptimizer/ExponentialDecay/Pow*
T0
B
optimizer/gradients/ShapeConst*
valueB *
dtype0
J
optimizer/gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0
u
optimizer/gradients/FillFilloptimizer/gradients/Shapeoptimizer/gradients/grad_ys_0*
T0*

index_type0
k
=optimizer/gradients/evalution_metrics/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0
²
7optimizer/gradients/evalution_metrics/Mean_grad/ReshapeReshapeoptimizer/gradients/Fill=optimizer/gradients/evalution_metrics/Mean_grad/Reshape/shape*
T0*
Tshape0

5optimizer/gradients/evalution_metrics/Mean_grad/ShapeShape)evalution_metrics/cross_entropy/Reshape_2*
T0*
out_type0
Ē
4optimizer/gradients/evalution_metrics/Mean_grad/TileTile7optimizer/gradients/evalution_metrics/Mean_grad/Reshape5optimizer/gradients/evalution_metrics/Mean_grad/Shape*

Tmultiples0*
T0

7optimizer/gradients/evalution_metrics/Mean_grad/Shape_1Shape)evalution_metrics/cross_entropy/Reshape_2*
T0*
out_type0
`
7optimizer/gradients/evalution_metrics/Mean_grad/Shape_2Const*
valueB *
dtype0
c
5optimizer/gradients/evalution_metrics/Mean_grad/ConstConst*
valueB: *
dtype0
Ņ
4optimizer/gradients/evalution_metrics/Mean_grad/ProdProd7optimizer/gradients/evalution_metrics/Mean_grad/Shape_15optimizer/gradients/evalution_metrics/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0
e
7optimizer/gradients/evalution_metrics/Mean_grad/Const_1Const*
valueB: *
dtype0
Ö
6optimizer/gradients/evalution_metrics/Mean_grad/Prod_1Prod7optimizer/gradients/evalution_metrics/Mean_grad/Shape_27optimizer/gradients/evalution_metrics/Mean_grad/Const_1*
T0*
	keep_dims( *

Tidx0
c
9optimizer/gradients/evalution_metrics/Mean_grad/Maximum/yConst*
value	B :*
dtype0
¾
7optimizer/gradients/evalution_metrics/Mean_grad/MaximumMaximum6optimizer/gradients/evalution_metrics/Mean_grad/Prod_19optimizer/gradients/evalution_metrics/Mean_grad/Maximum/y*
T0
¼
8optimizer/gradients/evalution_metrics/Mean_grad/floordivFloorDiv4optimizer/gradients/evalution_metrics/Mean_grad/Prod7optimizer/gradients/evalution_metrics/Mean_grad/Maximum*
T0

4optimizer/gradients/evalution_metrics/Mean_grad/CastCast8optimizer/gradients/evalution_metrics/Mean_grad/floordiv*

SrcT0*

DstT0
·
7optimizer/gradients/evalution_metrics/Mean_grad/truedivRealDiv4optimizer/gradients/evalution_metrics/Mean_grad/Tile4optimizer/gradients/evalution_metrics/Mean_grad/Cast*
T0

Hoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_2_grad/ShapeShapeevalution_metrics/cross_entropy*
T0*
out_type0
ļ
Joptimizer/gradients/evalution_metrics/cross_entropy/Reshape_2_grad/ReshapeReshape7optimizer/gradients/evalution_metrics/Mean_grad/truedivHoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_2_grad/Shape*
T0*
Tshape0
W
optimizer/gradients/zeros_like	ZerosLike!evalution_metrics/cross_entropy:1*
T0
z
Goptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0
ū
Coptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims
ExpandDimsJoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_2_grad/ReshapeGoptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims/dim*

Tdim0*
T0
“
<optimizer/gradients/evalution_metrics/cross_entropy_grad/mulMulCoptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims!evalution_metrics/cross_entropy:1*
T0

Coptimizer/gradients/evalution_metrics/cross_entropy_grad/LogSoftmax
LogSoftmax'evalution_metrics/cross_entropy/Reshape*
T0

<optimizer/gradients/evalution_metrics/cross_entropy_grad/NegNegCoptimizer/gradients/evalution_metrics/cross_entropy_grad/LogSoftmax*
T0
|
Ioptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims_1/dimConst*
valueB :
’’’’’’’’’*
dtype0
’
Eoptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims_1
ExpandDimsJoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_2_grad/ReshapeIoptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims_1/dim*

Tdim0*
T0
Ó
>optimizer/gradients/evalution_metrics/cross_entropy_grad/mul_1MulEoptimizer/gradients/evalution_metrics/cross_entropy_grad/ExpandDims_1<optimizer/gradients/evalution_metrics/cross_entropy_grad/Neg*
T0
Ń
Ioptimizer/gradients/evalution_metrics/cross_entropy_grad/tuple/group_depsNoOp=^optimizer/gradients/evalution_metrics/cross_entropy_grad/mul?^optimizer/gradients/evalution_metrics/cross_entropy_grad/mul_1
Į
Qoptimizer/gradients/evalution_metrics/cross_entropy_grad/tuple/control_dependencyIdentity<optimizer/gradients/evalution_metrics/cross_entropy_grad/mulJ^optimizer/gradients/evalution_metrics/cross_entropy_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimizer/gradients/evalution_metrics/cross_entropy_grad/mul
Ē
Soptimizer/gradients/evalution_metrics/cross_entropy_grad/tuple/control_dependency_1Identity>optimizer/gradients/evalution_metrics/cross_entropy_grad/mul_1J^optimizer/gradients/evalution_metrics/cross_entropy_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@optimizer/gradients/evalution_metrics/cross_entropy_grad/mul_1
w
Foptimizer/gradients/evalution_metrics/cross_entropy/Reshape_grad/ShapeShapemodel/Softmax*
T0*
out_type0

Hoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_grad/ReshapeReshapeQoptimizer/gradients/evalution_metrics/cross_entropy_grad/tuple/control_dependencyFoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_grad/Shape*
T0*
Tshape0

*optimizer/gradients/model/Softmax_grad/mulMulHoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_grad/Reshapemodel/Softmax*
T0
j
<optimizer/gradients/model/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0
Į
*optimizer/gradients/model/Softmax_grad/SumSum*optimizer/gradients/model/Softmax_grad/mul<optimizer/gradients/model/Softmax_grad/Sum/reduction_indices*
T0*
	keep_dims( *

Tidx0
i
4optimizer/gradients/model/Softmax_grad/Reshape/shapeConst*
valueB"’’’’   *
dtype0
²
.optimizer/gradients/model/Softmax_grad/ReshapeReshape*optimizer/gradients/model/Softmax_grad/Sum4optimizer/gradients/model/Softmax_grad/Reshape/shape*
T0*
Tshape0
“
*optimizer/gradients/model/Softmax_grad/subSubHoptimizer/gradients/evalution_metrics/cross_entropy/Reshape_grad/Reshape.optimizer/gradients/model/Softmax_grad/Reshape*
T0
w
,optimizer/gradients/model/Softmax_grad/mul_1Mul*optimizer/gradients/model/Softmax_grad/submodel/Softmax*
T0
\
*optimizer/gradients/model/add_4_grad/ShapeShapemodel/MatMul_1*
T0*
out_type0
Z
,optimizer/gradients/model/add_4_grad/Shape_1Const*
valueB:*
dtype0
¶
:optimizer/gradients/model/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs*optimizer/gradients/model/add_4_grad/Shape,optimizer/gradients/model/add_4_grad/Shape_1*
T0
æ
(optimizer/gradients/model/add_4_grad/SumSum,optimizer/gradients/model/Softmax_grad/mul_1:optimizer/gradients/model/add_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¤
,optimizer/gradients/model/add_4_grad/ReshapeReshape(optimizer/gradients/model/add_4_grad/Sum*optimizer/gradients/model/add_4_grad/Shape*
T0*
Tshape0
Ć
*optimizer/gradients/model/add_4_grad/Sum_1Sum,optimizer/gradients/model/Softmax_grad/mul_1<optimizer/gradients/model/add_4_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
Ŗ
.optimizer/gradients/model/add_4_grad/Reshape_1Reshape*optimizer/gradients/model/add_4_grad/Sum_1,optimizer/gradients/model/add_4_grad/Shape_1*
T0*
Tshape0

5optimizer/gradients/model/add_4_grad/tuple/group_depsNoOp-^optimizer/gradients/model/add_4_grad/Reshape/^optimizer/gradients/model/add_4_grad/Reshape_1
ł
=optimizer/gradients/model/add_4_grad/tuple/control_dependencyIdentity,optimizer/gradients/model/add_4_grad/Reshape6^optimizer/gradients/model/add_4_grad/tuple/group_deps*
T0*?
_class5
31loc:@optimizer/gradients/model/add_4_grad/Reshape
’
?optimizer/gradients/model/add_4_grad/tuple/control_dependency_1Identity.optimizer/gradients/model/add_4_grad/Reshape_16^optimizer/gradients/model/add_4_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/model/add_4_grad/Reshape_1
½
.optimizer/gradients/model/MatMul_1_grad/MatMulMatMul=optimizer/gradients/model/add_4_grad/tuple/control_dependencyweights/weight_5/read*
transpose_a( *
transpose_b(*
T0
µ
0optimizer/gradients/model/MatMul_1_grad/MatMul_1MatMulmodel/add_3=optimizer/gradients/model/add_4_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0
¤
8optimizer/gradients/model/MatMul_1_grad/tuple/group_depsNoOp/^optimizer/gradients/model/MatMul_1_grad/MatMul1^optimizer/gradients/model/MatMul_1_grad/MatMul_1

@optimizer/gradients/model/MatMul_1_grad/tuple/control_dependencyIdentity.optimizer/gradients/model/MatMul_1_grad/MatMul9^optimizer/gradients/model/MatMul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/model/MatMul_1_grad/MatMul

Boptimizer/gradients/model/MatMul_1_grad/tuple/control_dependency_1Identity0optimizer/gradients/model/MatMul_1_grad/MatMul_19^optimizer/gradients/model/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@optimizer/gradients/model/MatMul_1_grad/MatMul_1
Z
*optimizer/gradients/model/add_3_grad/ShapeShapemodel/MatMul*
T0*
out_type0
[
,optimizer/gradients/model/add_3_grad/Shape_1Const*
valueB:Č*
dtype0
¶
:optimizer/gradients/model/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs*optimizer/gradients/model/add_3_grad/Shape,optimizer/gradients/model/add_3_grad/Shape_1*
T0
Ó
(optimizer/gradients/model/add_3_grad/SumSum@optimizer/gradients/model/MatMul_1_grad/tuple/control_dependency:optimizer/gradients/model/add_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
¤
,optimizer/gradients/model/add_3_grad/ReshapeReshape(optimizer/gradients/model/add_3_grad/Sum*optimizer/gradients/model/add_3_grad/Shape*
T0*
Tshape0
×
*optimizer/gradients/model/add_3_grad/Sum_1Sum@optimizer/gradients/model/MatMul_1_grad/tuple/control_dependency<optimizer/gradients/model/add_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
Ŗ
.optimizer/gradients/model/add_3_grad/Reshape_1Reshape*optimizer/gradients/model/add_3_grad/Sum_1,optimizer/gradients/model/add_3_grad/Shape_1*
T0*
Tshape0

5optimizer/gradients/model/add_3_grad/tuple/group_depsNoOp-^optimizer/gradients/model/add_3_grad/Reshape/^optimizer/gradients/model/add_3_grad/Reshape_1
ł
=optimizer/gradients/model/add_3_grad/tuple/control_dependencyIdentity,optimizer/gradients/model/add_3_grad/Reshape6^optimizer/gradients/model/add_3_grad/tuple/group_deps*
T0*?
_class5
31loc:@optimizer/gradients/model/add_3_grad/Reshape
’
?optimizer/gradients/model/add_3_grad/tuple/control_dependency_1Identity.optimizer/gradients/model/add_3_grad/Reshape_16^optimizer/gradients/model/add_3_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/model/add_3_grad/Reshape_1
»
,optimizer/gradients/model/MatMul_grad/MatMulMatMul=optimizer/gradients/model/add_3_grad/tuple/control_dependencyweights/weight_4/read*
T0*
transpose_a( *
transpose_b(
°
.optimizer/gradients/model/MatMul_grad/MatMul_1MatMulmodel/LF=optimizer/gradients/model/add_3_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 

6optimizer/gradients/model/MatMul_grad/tuple/group_depsNoOp-^optimizer/gradients/model/MatMul_grad/MatMul/^optimizer/gradients/model/MatMul_grad/MatMul_1
ū
>optimizer/gradients/model/MatMul_grad/tuple/control_dependencyIdentity,optimizer/gradients/model/MatMul_grad/MatMul7^optimizer/gradients/model/MatMul_grad/tuple/group_deps*?
_class5
31loc:@optimizer/gradients/model/MatMul_grad/MatMul*
T0

@optimizer/gradients/model/MatMul_grad/tuple/control_dependency_1Identity.optimizer/gradients/model/MatMul_grad/MatMul_17^optimizer/gradients/model/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/model/MatMul_grad/MatMul_1
S
'optimizer/gradients/model/LF_grad/ShapeShapemodel/L3*
out_type0*
T0
“
)optimizer/gradients/model/LF_grad/ReshapeReshape>optimizer/gradients/model/MatMul_grad/tuple/control_dependency'optimizer/gradients/model/LF_grad/Shape*
T0*
Tshape0
t
*optimizer/gradients/model/L3_grad/ReluGradReluGrad)optimizer/gradients/model/LF_grad/Reshapemodel/L3*
T0
\
*optimizer/gradients/model/add_2_grad/ShapeShapemodel/Conv2D_2*
T0*
out_type0
Z
,optimizer/gradients/model/add_2_grad/Shape_1Const*
valueB:*
dtype0
¶
:optimizer/gradients/model/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs*optimizer/gradients/model/add_2_grad/Shape,optimizer/gradients/model/add_2_grad/Shape_1*
T0
½
(optimizer/gradients/model/add_2_grad/SumSum*optimizer/gradients/model/L3_grad/ReluGrad:optimizer/gradients/model/add_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¤
,optimizer/gradients/model/add_2_grad/ReshapeReshape(optimizer/gradients/model/add_2_grad/Sum*optimizer/gradients/model/add_2_grad/Shape*
T0*
Tshape0
Į
*optimizer/gradients/model/add_2_grad/Sum_1Sum*optimizer/gradients/model/L3_grad/ReluGrad<optimizer/gradients/model/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
Ŗ
.optimizer/gradients/model/add_2_grad/Reshape_1Reshape*optimizer/gradients/model/add_2_grad/Sum_1,optimizer/gradients/model/add_2_grad/Shape_1*
T0*
Tshape0

5optimizer/gradients/model/add_2_grad/tuple/group_depsNoOp-^optimizer/gradients/model/add_2_grad/Reshape/^optimizer/gradients/model/add_2_grad/Reshape_1
ł
=optimizer/gradients/model/add_2_grad/tuple/control_dependencyIdentity,optimizer/gradients/model/add_2_grad/Reshape6^optimizer/gradients/model/add_2_grad/tuple/group_deps*?
_class5
31loc:@optimizer/gradients/model/add_2_grad/Reshape*
T0
’
?optimizer/gradients/model/add_2_grad/tuple/control_dependency_1Identity.optimizer/gradients/model/add_2_grad/Reshape_16^optimizer/gradients/model/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@optimizer/gradients/model/add_2_grad/Reshape_1
{
.optimizer/gradients/model/Conv2D_2_grad/ShapeNShapeNmodel/L2weights/weight_3/read*
T0*
out_type0*
N
j
-optimizer/gradients/model/Conv2D_2_grad/ConstConst*%
valueB"            *
dtype0
Ļ
;optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropInputConv2DBackpropInput.optimizer/gradients/model/Conv2D_2_grad/ShapeNweights/weight_3/read=optimizer/gradients/model/add_2_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Ć
<optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/L2-optimizer/gradients/model/Conv2D_2_grad/Const=optimizer/gradients/model/add_2_grad/tuple/control_dependency*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
½
8optimizer/gradients/model/Conv2D_2_grad/tuple/group_depsNoOp<^optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropInput=^optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropFilter

@optimizer/gradients/model/Conv2D_2_grad/tuple/control_dependencyIdentity;optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropInput9^optimizer/gradients/model/Conv2D_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropInput
”
Boptimizer/gradients/model/Conv2D_2_grad/tuple/control_dependency_1Identity<optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropFilter9^optimizer/gradients/model/Conv2D_2_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimizer/gradients/model/Conv2D_2_grad/Conv2DBackpropFilter

*optimizer/gradients/model/L2_grad/ReluGradReluGrad@optimizer/gradients/model/Conv2D_2_grad/tuple/control_dependencymodel/L2*
T0
\
*optimizer/gradients/model/add_1_grad/ShapeShapemodel/Conv2D_1*
T0*
out_type0
Z
,optimizer/gradients/model/add_1_grad/Shape_1Const*
valueB:*
dtype0
¶
:optimizer/gradients/model/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs*optimizer/gradients/model/add_1_grad/Shape,optimizer/gradients/model/add_1_grad/Shape_1*
T0
½
(optimizer/gradients/model/add_1_grad/SumSum*optimizer/gradients/model/L2_grad/ReluGrad:optimizer/gradients/model/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0
¤
,optimizer/gradients/model/add_1_grad/ReshapeReshape(optimizer/gradients/model/add_1_grad/Sum*optimizer/gradients/model/add_1_grad/Shape*
T0*
Tshape0
Į
*optimizer/gradients/model/add_1_grad/Sum_1Sum*optimizer/gradients/model/L2_grad/ReluGrad<optimizer/gradients/model/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
Ŗ
.optimizer/gradients/model/add_1_grad/Reshape_1Reshape*optimizer/gradients/model/add_1_grad/Sum_1,optimizer/gradients/model/add_1_grad/Shape_1*
T0*
Tshape0

5optimizer/gradients/model/add_1_grad/tuple/group_depsNoOp-^optimizer/gradients/model/add_1_grad/Reshape/^optimizer/gradients/model/add_1_grad/Reshape_1
ł
=optimizer/gradients/model/add_1_grad/tuple/control_dependencyIdentity,optimizer/gradients/model/add_1_grad/Reshape6^optimizer/gradients/model/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@optimizer/gradients/model/add_1_grad/Reshape
’
?optimizer/gradients/model/add_1_grad/tuple/control_dependency_1Identity.optimizer/gradients/model/add_1_grad/Reshape_16^optimizer/gradients/model/add_1_grad/tuple/group_deps*A
_class7
53loc:@optimizer/gradients/model/add_1_grad/Reshape_1*
T0
{
.optimizer/gradients/model/Conv2D_1_grad/ShapeNShapeNmodel/L1weights/weight_2/read*
T0*
out_type0*
N
j
-optimizer/gradients/model/Conv2D_1_grad/ConstConst*%
valueB"            *
dtype0
Ļ
;optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropInputConv2DBackpropInput.optimizer/gradients/model/Conv2D_1_grad/ShapeNweights/weight_2/read=optimizer/gradients/model/add_1_grad/tuple/control_dependency*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
	dilations

Ć
<optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropFilterConv2DBackpropFiltermodel/L1-optimizer/gradients/model/Conv2D_1_grad/Const=optimizer/gradients/model/add_1_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
½
8optimizer/gradients/model/Conv2D_1_grad/tuple/group_depsNoOp<^optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropInput=^optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropFilter

@optimizer/gradients/model/Conv2D_1_grad/tuple/control_dependencyIdentity;optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropInput9^optimizer/gradients/model/Conv2D_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropInput
”
Boptimizer/gradients/model/Conv2D_1_grad/tuple/control_dependency_1Identity<optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropFilter9^optimizer/gradients/model/Conv2D_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@optimizer/gradients/model/Conv2D_1_grad/Conv2DBackpropFilter

*optimizer/gradients/model/L1_grad/ReluGradReluGrad@optimizer/gradients/model/Conv2D_1_grad/tuple/control_dependencymodel/L1*
T0
X
(optimizer/gradients/model/add_grad/ShapeShapemodel/Conv2D*
T0*
out_type0
X
*optimizer/gradients/model/add_grad/Shape_1Const*
valueB:*
dtype0
°
8optimizer/gradients/model/add_grad/BroadcastGradientArgsBroadcastGradientArgs(optimizer/gradients/model/add_grad/Shape*optimizer/gradients/model/add_grad/Shape_1*
T0
¹
&optimizer/gradients/model/add_grad/SumSum*optimizer/gradients/model/L1_grad/ReluGrad8optimizer/gradients/model/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0

*optimizer/gradients/model/add_grad/ReshapeReshape&optimizer/gradients/model/add_grad/Sum(optimizer/gradients/model/add_grad/Shape*
T0*
Tshape0
½
(optimizer/gradients/model/add_grad/Sum_1Sum*optimizer/gradients/model/L1_grad/ReluGrad:optimizer/gradients/model/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0
¤
,optimizer/gradients/model/add_grad/Reshape_1Reshape(optimizer/gradients/model/add_grad/Sum_1*optimizer/gradients/model/add_grad/Shape_1*
T0*
Tshape0

3optimizer/gradients/model/add_grad/tuple/group_depsNoOp+^optimizer/gradients/model/add_grad/Reshape-^optimizer/gradients/model/add_grad/Reshape_1
ń
;optimizer/gradients/model/add_grad/tuple/control_dependencyIdentity*optimizer/gradients/model/add_grad/Reshape4^optimizer/gradients/model/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@optimizer/gradients/model/add_grad/Reshape
÷
=optimizer/gradients/model/add_grad/tuple/control_dependency_1Identity,optimizer/gradients/model/add_grad/Reshape_14^optimizer/gradients/model/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@optimizer/gradients/model/add_grad/Reshape_1
}
,optimizer/gradients/model/Conv2D_grad/ShapeNShapeNinputs/inputweights/weight_1/read*
T0*
out_type0*
N
h
+optimizer/gradients/model/Conv2D_grad/ConstConst*%
valueB"            *
dtype0
É
9optimizer/gradients/model/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput,optimizer/gradients/model/Conv2D_grad/ShapeNweights/weight_1/read;optimizer/gradients/model/add_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
Į
:optimizer/gradients/model/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinputs/input+optimizer/gradients/model/Conv2D_grad/Const;optimizer/gradients/model/add_grad/tuple/control_dependency*
paddingSAME*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
·
6optimizer/gradients/model/Conv2D_grad/tuple/group_depsNoOp:^optimizer/gradients/model/Conv2D_grad/Conv2DBackpropInput;^optimizer/gradients/model/Conv2D_grad/Conv2DBackpropFilter

>optimizer/gradients/model/Conv2D_grad/tuple/control_dependencyIdentity9optimizer/gradients/model/Conv2D_grad/Conv2DBackpropInput7^optimizer/gradients/model/Conv2D_grad/tuple/group_deps*
T0*L
_classB
@>loc:@optimizer/gradients/model/Conv2D_grad/Conv2DBackpropInput

@optimizer/gradients/model/Conv2D_grad/tuple/control_dependency_1Identity:optimizer/gradients/model/Conv2D_grad/Conv2DBackpropFilter7^optimizer/gradients/model/Conv2D_grad/tuple/group_deps*
T0*M
_classC
A?loc:@optimizer/gradients/model/Conv2D_grad/Conv2DBackpropFilter

Foptimizer/GradientDescent/update_weights/weight_1/ApplyGradientDescentApplyGradientDescentweights/weight_1optimizer/ExponentialDecay@optimizer/gradients/model/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@weights/weight_1

Foptimizer/GradientDescent/update_weights/weight_2/ApplyGradientDescentApplyGradientDescentweights/weight_2optimizer/ExponentialDecayBoptimizer/gradients/model/Conv2D_1_grad/tuple/control_dependency_1*
T0*#
_class
loc:@weights/weight_2*
use_locking( 

Foptimizer/GradientDescent/update_weights/weight_3/ApplyGradientDescentApplyGradientDescentweights/weight_3optimizer/ExponentialDecayBoptimizer/gradients/model/Conv2D_2_grad/tuple/control_dependency_1*#
_class
loc:@weights/weight_3*
use_locking( *
T0

Foptimizer/GradientDescent/update_weights/weight_4/ApplyGradientDescentApplyGradientDescentweights/weight_4optimizer/ExponentialDecay@optimizer/gradients/model/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@weights/weight_4

Foptimizer/GradientDescent/update_weights/weight_5/ApplyGradientDescentApplyGradientDescentweights/weight_5optimizer/ExponentialDecayBoptimizer/gradients/model/MatMul_1_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@weights/weight_5

Coptimizer/GradientDescent/update_biases/bias_1/ApplyGradientDescentApplyGradientDescentbiases/bias_1optimizer/ExponentialDecay=optimizer/gradients/model/add_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@biases/bias_1

Coptimizer/GradientDescent/update_biases/bias_2/ApplyGradientDescentApplyGradientDescentbiases/bias_2optimizer/ExponentialDecay?optimizer/gradients/model/add_1_grad/tuple/control_dependency_1*
T0* 
_class
loc:@biases/bias_2*
use_locking( 

Coptimizer/GradientDescent/update_biases/bias_3/ApplyGradientDescentApplyGradientDescentbiases/bias_3optimizer/ExponentialDecay?optimizer/gradients/model/add_2_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@biases/bias_3

Coptimizer/GradientDescent/update_biases/bias_4/ApplyGradientDescentApplyGradientDescentbiases/bias_4optimizer/ExponentialDecay?optimizer/gradients/model/add_3_grad/tuple/control_dependency_1*
T0* 
_class
loc:@biases/bias_4*
use_locking( 

Coptimizer/GradientDescent/update_biases/bias_5/ApplyGradientDescentApplyGradientDescentbiases/bias_5optimizer/ExponentialDecay?optimizer/gradients/model/add_4_grad/tuple/control_dependency_1*
use_locking( *
T0* 
_class
loc:@biases/bias_5
ó
 optimizer/GradientDescent/updateNoOpG^optimizer/GradientDescent/update_weights/weight_1/ApplyGradientDescentG^optimizer/GradientDescent/update_weights/weight_2/ApplyGradientDescentG^optimizer/GradientDescent/update_weights/weight_3/ApplyGradientDescentG^optimizer/GradientDescent/update_weights/weight_4/ApplyGradientDescentG^optimizer/GradientDescent/update_weights/weight_5/ApplyGradientDescentD^optimizer/GradientDescent/update_biases/bias_1/ApplyGradientDescentD^optimizer/GradientDescent/update_biases/bias_2/ApplyGradientDescentD^optimizer/GradientDescent/update_biases/bias_3/ApplyGradientDescentD^optimizer/GradientDescent/update_biases/bias_4/ApplyGradientDescentD^optimizer/GradientDescent/update_biases/bias_5/ApplyGradientDescent

optimizer/GradientDescent/valueConst!^optimizer/GradientDescent/update*%
_class
loc:@optimizer/Variable*
value	B :*
dtype0

optimizer/GradientDescent	AssignAddoptimizer/Variableoptimizer/GradientDescent/value*
use_locking( *
T0*%
_class
loc:@optimizer/Variable
6
	loss/tagsConst*
dtype0*
valueB
 Bloss
A
lossScalarSummary	loss/tagsevalution_metrics/Mean*
T0
4
acc/tagsConst*
valueB	 Bacc*
dtype0
A
accScalarSummaryacc/tagsevalution_metrics/Mean_1*
T0
=
weight_1/tagConst*
dtype0*
valueB Bweight_1
J
weight_1HistogramSummaryweight_1/tagweights/weight_1/read*
T0
=
weight_2/tagConst*
valueB Bweight_2*
dtype0
J
weight_2HistogramSummaryweight_2/tagweights/weight_2/read*
T0
=
weight_3/tagConst*
valueB Bweight_3*
dtype0
J
weight_3HistogramSummaryweight_3/tagweights/weight_3/read*
T0
=
weight_4/tagConst*
valueB Bweight_4*
dtype0
J
weight_4HistogramSummaryweight_4/tagweights/weight_4/read*
T0
=
weight_5/tagConst*
valueB Bweight_5*
dtype0
J
weight_5HistogramSummaryweight_5/tagweights/weight_5/read*
T0
9

bias_1/tagConst*
valueB Bbias_1*
dtype0
C
bias_1HistogramSummary
bias_1/tagbiases/bias_1/read*
T0
9

bias_2/tagConst*
valueB Bbias_2*
dtype0
C
bias_2HistogramSummary
bias_2/tagbiases/bias_2/read*
T0
9

bias_3/tagConst*
valueB Bbias_3*
dtype0
C
bias_3HistogramSummary
bias_3/tagbiases/bias_3/read*
T0
9

bias_4/tagConst*
valueB Bbias_4*
dtype0
C
bias_4HistogramSummary
bias_4/tagbiases/bias_4/read*
T0
9

bias_5/tagConst*
valueB Bbias_5*
dtype0
C
bias_5HistogramSummary
bias_5/tagbiases/bias_5/read*
T0

Merge/MergeSummaryMergeSummarylossaccweight_1weight_2weight_3weight_4weight_5bias_1bias_2bias_3bias_4bias_5*
N

initNoOp^weights/weight_1/Assign^weights/weight_2/Assign^weights/weight_3/Assign^weights/weight_4/Assign^weights/weight_5/Assign^biases/bias_1/Assign^biases/bias_2/Assign^biases/bias_3/Assign^biases/bias_4/Assign^biases/bias_5/Assign^optimizer/Variable/Assign
8

save/ConstConst*
valueB Bmodel*
dtype0
’
save/SaveV2/tensor_namesConst*Ī
valueÄBĮBbiases/bias_1Bbiases/bias_2Bbiases/bias_3Bbiases/bias_4Bbiases/bias_5Boptimizer/VariableBweights/weight_1Bweights/weight_2Bweights/weight_3Bweights/weight_4Bweights/weight_5*
dtype0
]
save/SaveV2/shape_and_slicesConst*)
value BB B B B B B B B B B B *
dtype0
­
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbiases/bias_1biases/bias_2biases/bias_3biases/bias_4biases/bias_5optimizer/Variableweights/weight_1weights/weight_2weights/weight_3weights/weight_4weights/weight_5*
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const

save/RestoreV2/tensor_namesConst"/device:CPU:0*Ī
valueÄBĮBbiases/bias_1Bbiases/bias_2Bbiases/bias_3Bbiases/bias_4Bbiases/bias_5Boptimizer/VariableBweights/weight_1Bweights/weight_2Bweights/weight_3Bweights/weight_4Bweights/weight_5*
dtype0
o
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*)
value BB B B B B B B B B B B *
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2

save/AssignAssignbiases/bias_1save/RestoreV2*
use_locking(*
T0* 
_class
loc:@biases/bias_1*
validate_shape(

save/Assign_1Assignbiases/bias_2save/RestoreV2:1*
use_locking(*
T0* 
_class
loc:@biases/bias_2*
validate_shape(

save/Assign_2Assignbiases/bias_3save/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@biases/bias_3*
validate_shape(

save/Assign_3Assignbiases/bias_4save/RestoreV2:3*
use_locking(*
T0* 
_class
loc:@biases/bias_4*
validate_shape(

save/Assign_4Assignbiases/bias_5save/RestoreV2:4*
T0* 
_class
loc:@biases/bias_5*
validate_shape(*
use_locking(

save/Assign_5Assignoptimizer/Variablesave/RestoreV2:5*
use_locking(*
T0*%
_class
loc:@optimizer/Variable*
validate_shape(

save/Assign_6Assignweights/weight_1save/RestoreV2:6*
use_locking(*
T0*#
_class
loc:@weights/weight_1*
validate_shape(

save/Assign_7Assignweights/weight_2save/RestoreV2:7*
validate_shape(*
use_locking(*
T0*#
_class
loc:@weights/weight_2

save/Assign_8Assignweights/weight_3save/RestoreV2:8*
validate_shape(*
use_locking(*
T0*#
_class
loc:@weights/weight_3

save/Assign_9Assignweights/weight_4save/RestoreV2:9*
use_locking(*
T0*#
_class
loc:@weights/weight_4*
validate_shape(

save/Assign_10Assignweights/weight_5save/RestoreV2:10*
use_locking(*
T0*#
_class
loc:@weights/weight_5*
validate_shape(
Ē
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10"