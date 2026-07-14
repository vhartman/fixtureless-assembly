ur_base { multibody }

ur_floatX (ur_base){ joint:transX, limits:[-2 2], mass:.01 }
ur_floatY (ur_floatX){ joint:transY, limits:[-2 2], mass:.01 }
ur_floatZ (ur_floatY){ joint:transZ, limits:[0 3], mass:.01, q: 1 }
ur_floatRX (ur_floatZ){ joint:hingeX, limits:[-3.14159 3.14159], mass:.01 }
ur_floatRY (ur_floatRX){ joint:hingeY, limits:[-3.14159 3.14159], mass:.01 }
ur_ee_link (ur_floatRY){ joint:hingeZ, limits:[-3.14159 3.14159], mass:.01 }

ur_base_coll(ur_ee_link)   { shape:cylinder color:[1.,1.,1.,.5] size:[.06 .039] Q:"t(.0325 -.0 .0) d(90 0 1 0)", contact:-1  }

Prefix: "ur_"
Include: <../robotiq/robotiq.g>
Edit ur_robotiq_base (ur_ee_link) { Q:"d(90 0 1 0) t(-.0 -.0 .036)" }
