﻿#pragma checksum "..\..\..\UI\OptionCreateControl.xaml" "{8829d00f-11b8-4213-878b-770e8597ac16}" "7BD8C4C95971FBAE333BEDCA8789B8BA86782F776AD0F1ACC5479B03DDFD6AEA"
//------------------------------------------------------------------------------
// <auto-generated>
//     此代码由工具生成。
//     运行时版本:4.0.30319.42000
//
//     对此文件的更改可能会导致不正确的行为，并且如果
//     重新生成代码，这些更改将会丢失。
// </auto-generated>
//------------------------------------------------------------------------------

using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Ink;
using System.Windows.Input;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Media.TextFormatting;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Windows.Shell;
using zdhsys.Control;
using zdhsys.UI;


namespace zdhsys.UI {
    
    
    /// <summary>
    /// OptionCreateControl
    /// </summary>
    public partial class OptionCreateControl : System.Windows.Controls.UserControl, System.Windows.Markup.IComponentConnector {
        
        
        #line 33 "..\..\..\UI\OptionCreateControl.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal zdhsys.Control.DevTextBox txt_option;
        
        #line default
        #line hidden
        
        
        #line 34 "..\..\..\UI\OptionCreateControl.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal zdhsys.Control.DevButton1 Btn_Add;
        
        #line default
        #line hidden
        
        
        #line 35 "..\..\..\UI\OptionCreateControl.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal zdhsys.Control.DevButton1 Btn_Save;
        
        #line default
        #line hidden
        
        
        #line 36 "..\..\..\UI\OptionCreateControl.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal zdhsys.Control.DevButton1 Btn_Close;
        
        #line default
        #line hidden
        
        
        #line 44 "..\..\..\UI\OptionCreateControl.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Grid gd_header;
        
        #line default
        #line hidden
        
        
        #line 56 "..\..\..\UI\OptionCreateControl.xaml"
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1823:AvoidUnusedPrivateFields")]
        internal System.Windows.Controls.Grid gd_content;
        
        #line default
        #line hidden
        
        private bool _contentLoaded;
        
        /// <summary>
        /// InitializeComponent
        /// </summary>
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        public void InitializeComponent() {
            if (_contentLoaded) {
                return;
            }
            _contentLoaded = true;
            System.Uri resourceLocater = new System.Uri("/AI微纳实验软件;component/ui/optioncreatecontrol.xaml", System.UriKind.Relative);
            
            #line 1 "..\..\..\UI\OptionCreateControl.xaml"
            System.Windows.Application.LoadComponent(this, resourceLocater);
            
            #line default
            #line hidden
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal System.Delegate _CreateDelegate(System.Type delegateType, string handler) {
            return System.Delegate.CreateDelegate(delegateType, this, handler);
        }
        
        [System.Diagnostics.DebuggerNonUserCodeAttribute()]
        [System.CodeDom.Compiler.GeneratedCodeAttribute("PresentationBuildTasks", "4.0.0.0")]
        [System.ComponentModel.EditorBrowsableAttribute(System.ComponentModel.EditorBrowsableState.Never)]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Design", "CA1033:InterfaceMethodsShouldBeCallableByChildTypes")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Maintainability", "CA1502:AvoidExcessiveComplexity")]
        [System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1800:DoNotCastUnnecessarily")]
        void System.Windows.Markup.IComponentConnector.Connect(int connectionId, object target) {
            switch (connectionId)
            {
            case 1:
            
            #line 8 "..\..\..\UI\OptionCreateControl.xaml"
            ((zdhsys.UI.OptionCreateControl)(target)).Loaded += new System.Windows.RoutedEventHandler(this.UserControl_Loaded);
            
            #line default
            #line hidden
            return;
            case 2:
            this.txt_option = ((zdhsys.Control.DevTextBox)(target));
            return;
            case 3:
            this.Btn_Add = ((zdhsys.Control.DevButton1)(target));
            return;
            case 4:
            this.Btn_Save = ((zdhsys.Control.DevButton1)(target));
            return;
            case 5:
            this.Btn_Close = ((zdhsys.Control.DevButton1)(target));
            return;
            case 6:
            this.gd_header = ((System.Windows.Controls.Grid)(target));
            return;
            case 7:
            this.gd_content = ((System.Windows.Controls.Grid)(target));
            return;
            }
            this._contentLoaded = true;
        }
    }
}

