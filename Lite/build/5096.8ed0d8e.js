"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[5096,5971],{15971:(e,t,r)=>{r.r(t),r.d(t,{default:()=>v});var n=r(67504),a=r(2814),o=r(17543),c=r(60192),i=r(49005),s=r(15786),l=r(4265),d=r(90901),g=r(985),h=r(70942);class u{constructor(){this.matches=[],this.currentMatchIndex=null,this.isReadOnly=!0,this._changed=new h.Signal(this)}static canSearchOn(e){return e instanceof g.DocumentWidget&&e.content instanceof o.CSVViewer}getInitialQuery(e){return null}async startQuery(e,t){return this._target=t,this._query=e,t.content.searchService.find(e),this.matches}async endQuery(){this._target.content.searchService.clear()}async endSearch(){this._target.content.searchService.clear()}async highlightNext(){this._target.content.searchService.find(this._query)}async highlightPrevious(){this._target.content.searchService.find(this._query,!0)}async replaceCurrentMatch(e){return!1}async replaceAllMatches(e){return!1}get changed(){return this._changed}}const T="CSVTable",y="TSVTable",C={activate:function(e,t,r,n,c,i,s,l){let d;l&&(l.registerFactory(T,"delimiter",(e=>new o.CSVDelimiter({widget:e.content,translator:t}))),s&&(d=(0,a.createToolbarFactory)(l,s,T,C.id,t)));const g=new o.CSVViewerFactory({name:T,fileTypes:["csv"],defaultFor:["csv"],readOnly:!0,toolbarFactory:d,translator:t}),h=new a.WidgetTracker({namespace:"csvviewer"});let y=L.LIGHT_STYLE,_=L.LIGHT_TEXT_CONFIG;r&&r.restore(h,{command:"docmanager:open",args:e=>({path:e.context.path,factory:T}),name:e=>e.context.path}),e.docRegistry.addWidgetFactory(g);const v=e.docRegistry.getFileType("csv");g.widgetCreated.connect(((e,t)=>{h.add(t),t.context.pathChanged.connect((()=>{h.save(t)})),v&&(t.title.icon=v.icon,t.title.iconClass=v.iconClass,t.title.iconLabel=v.iconLabel),t.content.style=y,t.content.rendererConfig=_}));n&&n.themeChanged.connect((()=>{const e=!n||!n.theme||n.isLight(n.theme);y=e?L.LIGHT_STYLE:L.DARK_STYLE,_=e?L.LIGHT_TEXT_CONFIG:L.DARK_TEXT_CONFIG,h.forEach((e=>{e.content.style=y,e.content.rendererConfig=_}))})),c&&F(c,h,t),i&&i.register("csv",u)},id:"@jupyterlab/csvviewer-extension:csv",requires:[l.ITranslator],optional:[n.ILayoutRestorer,a.IThemeManager,i.IMainMenu,c.ISearchProviderRegistry,s.ISettingRegistry,a.IToolbarWidgetRegistry],autoStart:!0},_={activate:function(e,t,r,n,c,i,s,l){let d;l&&(l.registerFactory(y,"delimiter",(e=>new o.CSVDelimiter({widget:e.content,translator:t}))),s&&(d=(0,a.createToolbarFactory)(l,s,y,_.id,t)));const g=new o.TSVViewerFactory({name:y,fileTypes:["tsv"],defaultFor:["tsv"],readOnly:!0,toolbarFactory:d,translator:t}),h=new a.WidgetTracker({namespace:"tsvviewer"});let T=L.LIGHT_STYLE,C=L.LIGHT_TEXT_CONFIG;r&&r.restore(h,{command:"docmanager:open",args:e=>({path:e.context.path,factory:y}),name:e=>e.context.path}),e.docRegistry.addWidgetFactory(g);const v=e.docRegistry.getFileType("tsv");g.widgetCreated.connect(((e,t)=>{h.add(t),t.context.pathChanged.connect((()=>{h.save(t)})),v&&(t.title.icon=v.icon,t.title.iconClass=v.iconClass,t.title.iconLabel=v.iconLabel),t.content.style=T,t.content.rendererConfig=C}));n&&n.themeChanged.connect((()=>{const e=!n||!n.theme||n.isLight(n.theme);T=e?L.LIGHT_STYLE:L.DARK_STYLE,C=e?L.LIGHT_TEXT_CONFIG:L.DARK_TEXT_CONFIG,h.forEach((e=>{e.content.style=T,e.content.rendererConfig=C}))})),c&&F(c,h,t),i&&i.register("tsv",u)},id:"@jupyterlab/csvviewer-extension:tsv",requires:[l.ITranslator],optional:[n.ILayoutRestorer,a.IThemeManager,i.IMainMenu,c.ISearchProviderRegistry,s.ISettingRegistry,a.IToolbarWidgetRegistry],autoStart:!0};function F(e,t,r){const n=r.load("jupyterlab");e.editMenu.goToLiners.add({tracker:t,goToLine:e=>a.InputDialog.getNumber({title:n.__("Go to Line"),value:0}).then((t=>{t.button.accept&&null!==t.value&&e.content.goToLine(t.value)}))})}const v=[C,_];var L;!function(e){e.LIGHT_STYLE=Object.assign(Object.assign({},d.DataGrid.defaultStyle),{voidColor:"#F3F3F3",backgroundColor:"white",headerBackgroundColor:"#EEEEEE",gridLineColor:"rgba(20, 20, 20, 0.15)",headerGridLineColor:"rgba(20, 20, 20, 0.25)",rowBackgroundColor:e=>e%2==0?"#F5F5F5":"white"}),e.DARK_STYLE=Object.assign(Object.assign({},d.DataGrid.defaultStyle),{voidColor:"black",backgroundColor:"#111111",headerBackgroundColor:"#424242",gridLineColor:"rgba(235, 235, 235, 0.15)",headerGridLineColor:"rgba(235, 235, 235, 0.25)",rowBackgroundColor:e=>e%2==0?"#212121":"#111111"}),e.LIGHT_TEXT_CONFIG={textColor:"#111111",matchBackgroundColor:"#FFFFE0",currentMatchBackgroundColor:"#FFFF00",horizontalAlignment:"right"},e.DARK_TEXT_CONFIG={textColor:"#F5F5F5",matchBackgroundColor:"#838423",currentMatchBackgroundColor:"#A3807A",horizontalAlignment:"right"}}(L||(L={}))}}]);
//# sourceMappingURL=5096.8ed0d8e.js.map