   M S B A M L   `   `   `  ���� O  KAI微纳实验软件, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null+clr-namespace:zdhsys.Popupzdhsys.Popup  /clr-namespace:zdhsys.Controlzdhsys.Control      zdhsys.Popup.AddOptionNew   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     @d2http://schemas.microsoft.com/expression/blend/2008     Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     ,localclr-namespace:zdhsys.Popup     /local1clr-namespace:zdhsys.Control     -     � WindowStyle$
  None=� � AllowsTransparency . 
������  � Title$ 	AddOption��$	��750��$
��1300��.��� ���	#80000000.+�� $��Center=�$��Center=�$
��1300��$	��750��
������� � ColumnDefinitions �� $��80 ��� $��20 ��� $	��10* � � RowDefinitions �� $��40 ��� $��40 ��� $	��160 ��� .+��� $��0��$��0��$��10,0,0,0q�$��配方管理��$��15(����#999999$��Center=�   NoBorderButtonStyle0#��[  �� $��0��$��1��$��>��$��15(����#999999$��Center=��� $��0��$��2��$��创建配方��$��15(����#999999$��Center=��� $��1��$��0��$��10,0,0,0q�$��配方名称��$��15(����#000000$��Center=�� $��1��$��2�� �� $	��210 ��� $��40 ��� $��90 ��� $	��130 ��� $	��130 ��� $	��130 ��� $��* �.+�   zdhsys.Control.DevTextBox  -     Name$ 
txt_option��$	��200��$
��Left=��� $��1��$��状态��$��15(����#000000$��Right=�$��Center=��� -    ��Name$ cb��$��2��$��Center=�$��30��$��80��$
��Left=�
������    zdhsys.Control.DevButton1  -     Name$ Btn_Save��$��5��$	��130��$
��Left=�$��40��  -   $ 	Btn_Close��$��6��$	��130��$
��Left=�$��40��� $��2��$��3�����#F6F6F6.+�� -    �Name$ 	gd_header���� $��3��$��3��.��� -   $ 
gd_content��