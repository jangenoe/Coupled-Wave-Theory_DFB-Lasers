"use strict";(self.webpackChunk_JUPYTERLAB_CORE_OUTPUT=self.webpackChunk_JUPYTERLAB_CORE_OUTPUT||[]).push([[5238],{12563:(e,t,n)=>{n.d(t,{Z:()=>a});var i=n(34663),o=n.n(i),s=n(7638),r=n.n(s)()(o());r.push([e.id,"/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n","",{version:3,sources:["webpack://./../packages/application-extension/style/base.css"],names:[],mappings:"AAAA;;;8EAG8E",sourcesContent:["/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n"],sourceRoot:""}]);const a=r},59988:(e,t,n)=>{n.d(t,{Z:()=>a});var i=n(34663),o=n.n(i),s=n(7638),r=n.n(s)()(o());r.push([e.id,"/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n\n.jp-IFrameContainer iframe,\n.jp-IFrameContainer body {\n  margin: 0;\n  padding: 0;\n  overflow: hidden;\n  border: none;\n}\n","",{version:3,sources:["webpack://./../packages/iframe-extension/style/base.css"],names:[],mappings:"AAAA;;;8EAG8E;;AAE9E;;EAEE,SAAS;EACT,UAAU;EACV,gBAAgB;EAChB,YAAY;AACd",sourcesContent:["/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n\n.jp-IFrameContainer iframe,\n.jp-IFrameContainer body {\n  margin: 0;\n  padding: 0;\n  overflow: hidden;\n  border: none;\n}\n"],sourceRoot:""}]);const a=r},228:(e,t,n)=>{n.d(t,{Z:()=>a});var i=n(34663),o=n.n(i),s=n(7638),r=n.n(s)()(o());r.push([e.id,"/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n","",{version:3,sources:["webpack://./../packages/retro-application-extension/style/base.css"],names:[],mappings:"AAAA;;;8EAG8E",sourcesContent:["/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n"],sourceRoot:""}]);const a=r},94298:(e,t,n)=>{n.d(t,{Z:()=>a});var i=n(34663),o=n.n(i),s=n(7638),r=n.n(s)()(o());r.push([e.id,"/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n","",{version:3,sources:["webpack://./../packages/server-extension/style/base.css"],names:[],mappings:"AAAA;;;8EAG8E",sourcesContent:["/*-----------------------------------------------------------------------------\n| Copyright (c) Jupyter Development Team.\n| Distributed under the terms of the Modified BSD License.\n|----------------------------------------------------------------------------*/\n"],sourceRoot:""}]);const a=r},77600:(e,t,n)=>{n.r(t),n.d(t,{main:()=>A});var i=n(6623),o=n(55941);n(4436);const s=[n.e(9683).then(n.t.bind(n,79683,23))],r=[n.e(9151).then(n.t.bind(n,19151,23)),n.e(3946).then(n.t.bind(n,23946,23)),n.e(9265).then(n.t.bind(n,29265,23)),n.e(126).then(n.t.bind(n,30126,23))];async function a(e,t){try{return(await window._JUPYTERLAB[e].get(t))()}catch(n){throw console.warn(`Failed to create module: package: ${e}; module: ${t}`),n}}async function A(){const e=await Promise.all(r);let t=[n(69851),n(36627),n(88394).default.filter((({id:e})=>!["@retrolab/application-extension:logo","@retrolab/application-extension:opener"].includes(e))),n(31161),n(61474),n(29507).default.filter((({id:e})=>["@jupyterlab/application-extension:commands","@jupyterlab/application-extension:context-menu","@jupyterlab/application-extension:faviconbusy"].includes(e))),n(10310).default.filter((({id:e})=>["@jupyterlab/apputils-extension:palette","@jupyterlab/apputils-extension:settings","@jupyterlab/apputils-extension:state","@jupyterlab/apputils-extension:themes","@jupyterlab/apputils-extension:themes-palette-menu","@jupyterlab/apputils-extension:toolbar-registry"].includes(e))),n(15802).default.filter((({id:e})=>["@jupyterlab/codemirror-extension:services","@jupyterlab/codemirror-extension:codemirror"].includes(e))),n(65540).default.filter((({id:e})=>["@jupyterlab/completer-extension:manager"].includes(e))),n(78042),n(22437).default.filter((({id:e})=>["@jupyterlab/docmanager-extension:plugin","@jupyterlab/docmanager-extension:manager"].includes(e))),n(64948).default.filter((({id:e})=>["@jupyterlab/filebrowser-extension:factory"].includes(e))),n(46813),n(53516),n(62570).default.filter((({id:e})=>["@jupyterlab/notebook-extension:factory","@jupyterlab/notebook-extension:tracker","@jupyterlab/notebook-extension:widget-factory"].includes(e))),n(99256),n(89985),n(78036),n(85687),n(73624)];switch(o.PageConfig.getOption("retroPage")){case"tree":t=t.concat([n(64948).default.filter((({id:e})=>["@jupyterlab/filebrowser-extension:browser","@jupyterlab/filebrowser-extension:file-upload-status","@jupyterlab/filebrowser-extension:open-with"].includes(e))),n(24918).default.filter((({id:e})=>"@retrolab/tree-extension:new-terminal"!==e))]);break;case"notebooks":t=t.concat([n(48218),n(65540).default.filter((({id:e})=>["@jupyterlab/completer-extension:notebooks"].includes(e))),n(66325).default.filter((({id:e})=>["@jupyterlab/tooltip-extension:manager","@jupyterlab/tooltip-extension:notebooks"].includes(e)))]);break;case"consoles":t=t.concat([n(65540).default.filter((({id:e})=>["@jupyterlab/completer-extension:consoles"].includes(e))),n(66325).default.filter((({id:e})=>["@jupyterlab/tooltip-extension:manager","@jupyterlab/tooltip-extension:consoles"].includes(e)))]);break;case"edit":t=t.concat([n(65540).default.filter((({id:e})=>["@jupyterlab/completer-extension:files"].includes(e))),n(19656).default.filter((({id:e})=>["@jupyterlab/fileeditor-extension:plugin"].includes(e))),n(64948).default.filter((({id:e})=>["@jupyterlab/filebrowser-extension:browser"].includes(e)))])}const A=[],l=[],d=[],c=[],p=[],u=[],f=JSON.parse(o.PageConfig.getOption("federated_extensions")),m=new Set;function*g(e){let t;t=e.hasOwnProperty("__esModule")?e.default:e;let n=Array.isArray(t)?t:[t];for(let e of n)o.PageConfig.Extension.isDisabled(e.id)||(yield e)}f.forEach((e=>{e.liteExtension?u.push(a(e.name,e.extension)):(e.extension&&(m.add(e.name),l.push(a(e.name,e.extension))),e.mimeExtension&&(m.add(e.name),d.push(a(e.name,e.mimeExtension))),e.style&&c.push(a(e.name,e.style)))})),(await Promise.all(t)).forEach((e=>{for(let t of g(e))A.push(t)})),(await Promise.allSettled(d)).forEach((t=>{if("fulfilled"===t.status)for(let n of g(t.value))e.push(n);else console.error(t.reason)})),(await Promise.allSettled(l)).forEach((e=>{if("fulfilled"===e.status)for(let t of g(e.value))A.push(t);else console.error(e.reason)})),(await Promise.all(s)).forEach((e=>{for(let t of g(e))p.push(t)})),(await Promise.allSettled(u)).forEach((e=>{if("fulfilled"===e.status)for(let t of g(e.value))p.push(t);else console.error(e.reason)}));const b=new i.JupyterLiteServer({});b.registerPluginModules(p),await b.start();const{serviceManager:y}=b,{RetroApp:h}=n(95191),w=new h({serviceManager:y,mimeExtensions:e});w.name=o.PageConfig.getOption("appName")||"RetroLite",w.registerPluginModules(A),"true"===(o.PageConfig.getOption("exposeAppInBrowser")||"").toLowerCase()&&(window.jupyterapp=w),await w.start(),await w.restored}},4436:(e,t,n)=>{n.r(t),n(49336),n(51642),n(91778),n(40705),n(63505),n(44697),n(9123),n(45010),n(38786),n(47867),n(76294),n(70453),n(53974),n(10389),n(84396),n(88005),n(38016),n(83608),n(91487),n(74101);var i=n(1892),o=n.n(i),s=n(95760),r=n.n(s),a=n(38311),A=n.n(a),l=n(58192),d=n.n(l),c=n(38060),p=n.n(c),u=n(54865),f=n.n(u),m=n(12563),g={};g.styleTagTransform=f(),g.setAttributes=d(),g.insert=A().bind(null,"head"),g.domAPI=r(),g.insertStyleElement=p(),o()(m.Z,g),m.Z&&m.Z.locals&&m.Z.locals,n(47317);var b=n(59988),y={};y.styleTagTransform=f(),y.setAttributes=d(),y.insert=A().bind(null,"head"),y.domAPI=r(),y.insertStyleElement=p(),o()(b.Z,y),b.Z&&b.Z.locals&&b.Z.locals;var h=n(228),w={};w.styleTagTransform=f(),w.setAttributes=d(),w.insert=A().bind(null,"head"),w.domAPI=r(),w.insertStyleElement=p(),o()(h.Z,w),h.Z&&h.Z.locals&&h.Z.locals;var x=n(94298),v={};v.styleTagTransform=f(),v.setAttributes=d(),v.insert=A().bind(null,"head"),v.domAPI=r(),v.insertStyleElement=p(),o()(x.Z,v),x.Z&&x.Z.locals&&x.Z.locals,n(72073),n(96587),n(90381),n(28586),n(40250)},7413:e=>{e.exports="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII="},6196:e=>{e.exports="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII="},65767:e=>{e.exports="data:image/svg+xml,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3e%3cg fill=%27%235C7080%27%3e%3ccircle cx=%272%27 cy=%278.03%27 r=%272%27/%3e%3ccircle cx=%2714%27 cy=%278.03%27 r=%272%27/%3e%3ccircle cx=%278%27 cy=%278.03%27 r=%272%27/%3e%3c/g%3e%3c/svg%3e"},91116:e=>{e.exports="data:image/svg+xml,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3e%3cpath fill-rule=%27evenodd%27 clip-rule=%27evenodd%27 d=%27M10.71 7.29l-4-4a1.003 1.003 0 00-1.42 1.42L8.59 8 5.3 11.29c-.19.18-.3.43-.3.71a1.003 1.003 0 001.71.71l4-4c.18-.18.29-.43.29-.71 0-.28-.11-.53-.29-.71z%27 fill=%27%235C7080%27/%3e%3c/svg%3e"},83678:e=>{e.exports="data:image/svg+xml,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3e%3cpath fill-rule=%27evenodd%27 clip-rule=%27evenodd%27 d=%27M11 7H5c-.55 0-1 .45-1 1s.45 1 1 1h6c.55 0 1-.45 1-1s-.45-1-1-1z%27 fill=%27white%27/%3e%3c/svg%3e"},79080:e=>{e.exports="data:image/svg+xml,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 16 16%27%3e%3cpath fill-rule=%27evenodd%27 clip-rule=%27evenodd%27 d=%27M12 5c-.28 0-.53.11-.71.29L7 9.59l-2.29-2.3a1.003 1.003 0 00-1.42 1.42l3 3c.18.18.43.29.71.29s.53-.11.71-.29l5-5A1.003 1.003 0 0012 5z%27 fill=%27white%27/%3e%3c/svg%3e"}}]);
//# sourceMappingURL=5238.ba011ff.js.map