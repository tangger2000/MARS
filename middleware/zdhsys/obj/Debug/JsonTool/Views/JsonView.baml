   M S B A M L   `   `   `  ����O  KAI微纳实验软件, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null9!clr-namespace:JsonTool.ConvertersJsonTool.Converters  5      C&clr-namespace:JsonTool.Converters.JsonJsonTool.Converters.Json  5      9!clr-namespace:JsonTool.BehavioursJsonTool.Behaviours  5
      W SNewtonsoft.Json, Version=13.0.0.0, Culture=neutral, PublicKeyToken=30ad4fe6b2a6aeedT;clr-namespace:Newtonsoft.Json.Linq;assembly=Newtonsoft.JsonNewtonsoft.Json.Linq 5      5clr-namespace:JsonTool.ControlsJsonTool.Controls  5          JsonTool.Views.JsonView   5      R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5      R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5      Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5      @d2http://schemas.microsoft.com/expression/blend/2008     5      /w!clr-namespace:JsonTool.Converters     5      5jc&clr-namespace:JsonTool.Converters.Json     5      _ [Microsoft.Xaml.Behaviors, Version=1.1.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a3i+http://schemas.microsoft.com/xaml/behaviors  5      ^ ZICSharpCode.AvalonEdit, Version=6.3.0.90, Culture=neutral, PublicKeyToken=9cc39be672370310?
avalonedit.http://icsharpcode.net/sharpdevelop/avalonedit  5	      1	behaviour!clr-namespace:JsonTool.Behaviours  5
      Gjlinq;clr-namespace:Newtonsoft.Json.Linq;assembly=Newtonsoft.Json  5      (Aiclr-namespace:JsonTool.Controls  5      -   5          >�Name$  jsonView��5       ���#F2F2F2 � 	Resources 5       �� 
TargetType   TreeViewDefault01 �� BasedOn  TreeViewItemDefault010  *JsonTool.Converters.Json.JObjectToChildren  JObjectToChildren2  ,JsonTool.Converters.Json.JPropertyToChildren  JPropertyToChildren4  .JsonTool.Converters.Json.JPropertyNameToString  JPropertyNameToString5  /JsonTool.Converters.Json.JPropertyValueToString  JPropertyValueToString9  3JsonTool.Converters.Json.BoolToJsonSyntaxHighliting  TrueToHighliting0   *JsonTool.Converters.MethodToValueConverter  MethodToValueConverter'   !JsonTool.Converters.NegateBoolean  NegateBoolean  	 BooleanToVisibilityConverter �� DataType!  Newtonsoft.Json.Linq.JArray �� ItemsSource �� 	Converter �� ConverterParameter$	  Newtonsoft.Json.Linq.JProperty �� Orientation	 � ContextMenu
 m� CommandParameter m� Command �� Source%
  System.Windows.Markup.Reference"  Newtonsoft.Json.Linq.JObject �� Path�� 5   
   %$  'R�       7   'Q� "     7  &	 D     &	 W     &	 j     &	 }     &	 �     &	 �     &	 �     &		 �     (�� �     5!   '   *, +)7  (�� J    5%   '   *,	 +)7  7  7  (�� �    57   '   *, +)7  �� 5      " R�6   8   6*   �� 5      " Q�6   8   6.     5      6?     5      6C     5      6G     5      6I     5      6G     5      6'     5      6   �� 5      6+   �� 5!      "  6'    6H   �� 8   $ Children��.R�5"      �� $��Array��6   5#      5$      �� 5%      " 	 6'    6K   �� 8   .R�5&      �� $ 
Horizontal=�6   	 5'      �� 5(      .@�5)      m� $��
Edit value��6'   
 6;   ��  5*   '   �� *+)DataContext.EditValueFromJPropertyCommand+ 
  *
jsonView+�� 5+      6*   m� 5,      $��
Copy value��6'   
 6;   ��  5-   '   �� *+)DataContext.CopyValueFromJPropertyCommand+ 
  *
jsonView+m� 5.      $��	Copy name��6'   
 6:   ��  5/   '   �� **(DataContext.CopyNameFromJPropertyCommand+ 
  *
jsonView+50      .+�52      �� 
����  ���62   �� 8  �� 53      
����  �$	�� : ��62   �� 54      ��6!   �� 8  55      56      �� 57      "  6'    6I   �� 8   .R�58      �� ��6   �� $
 Path�69   59      5I      .��5K      �  � RowDefinitions 5L   
   �� 5M      $
��Auto �6   �� 5N      .+�5U   
   a� $��0��6   $��2��6%   .+�5V      `�  `� IsLocked . .�5a      c� .@�5b      �� $��0,0,10,0q�6    $��18��62   y�6>   ��  �� ElementName$ jsonView��$ ShowFileToolBar�# [	     JsonTool.Controls.AiButton  5c      -   5          Name$ 	btnFormat��5c   "    � ToolTip$ 	格式化��65   $��0,1,10,0q�6C   $��14��6U    
 
ButtonPath#��� 
 5d   "     Icon_Format   PathData# �  6G     BtnBlueBackground   	PathColor# �  6p   ��5c   `   �� *JsoFormatCommand+5d   p   5h      5o      5q      '  !ICSharpCode.AvalonEdit.TextEditor  5s   
   -   5          Name$ 
jsonEditor��5s   -   $��1��6       
IsReadOnly .  "   VerticalScrollBarVisibility$
 Auto=�6T   $��Stretch=�6x   $��0,0,5,0q�6�   $   HorizontalScrollBarVisibility$
 Auto=�5t   z      ShowLineNumbers . $��Consola)�5u   �      SyntaxHighlighting 5t   #   �� *
IsEditMode+# [ {�5u   #   �� *IsBusy+# [    Text. 6�   *  $Microsoft.Xaml.Behaviors.Interaction   	Behaviors 5v      -   'JsonTool.Behaviours.AvalonEditBehaviour  5w         GiveMeTheText 60   �� *
JsonString+ �� Mode$ TwoWay=� �� UpdateSourceTrigger$ PropertyChanged=�. 5x   '   5y      5�      5�      