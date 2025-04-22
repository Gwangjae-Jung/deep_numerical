resolutions_2D="16 32 64"
for i in $resolutions_2D
do
    python script__FSM_general.py -dim 2 --min_t 0.0 --max_t 10.0 --delta_t 0.1 --vhs_alpha 0.0 -prob bkw -res $i
done


resolutions_3D="8 16 32"
for i in $resolutions_3D
do
    python script__FSM_general.py -dim 3 --min_t 5.5 --max_t 10.0 --delta_t 0.1 --vhs_alpha 0.0 -prob bkw --quad_order_lebedev 5 -res $i
done