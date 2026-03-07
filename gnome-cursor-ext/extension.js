/**
 * AI4 Cursor Position — GNOME Shell Extension
 *
 * Registers a tiny DBus service at org.ai4.CursorPosition on the session bus.
 * Exposes one method: GetPosition() → (ii) returning [x, y] of the pointer.
 */
import GLib from "gi://GLib";
import Gio from "gi://Gio";
import Shell from "gi://Shell";
import { Extension } from "resource:///org/gnome/shell/extensions/extension.js";

const IFACE_XML = `
<node>
  <interface name="org.ai4.CursorPosition">
    <method name="GetPosition">
      <arg type="i" direction="out" name="x"/>
      <arg type="i" direction="out" name="y"/>
    </method>
    <method name="GetScreenSize">
      <arg type="i" direction="out" name="width"/>
      <arg type="i" direction="out" name="height"/>
    </method>
  </interface>
</node>`;

export default class CursorPositionExtension extends Extension {
    _dbusId = null;
    _nameId = null;

    enable() {
        const nodeInfo = Gio.DBusNodeInfo.new_for_xml(IFACE_XML);
        const ifaceInfo = nodeInfo.interfaces[0];

        this._dbusId = Gio.DBus.session.register_object(
            "/org/ai4/CursorPosition",
            ifaceInfo,
            (connection, sender, objectPath, interfaceName, methodName, parameters, invocation) => {
                if (methodName === "GetPosition") {
                    const [x, y] = global.get_pointer();
                    invocation.return_value(new GLib.Variant("(ii)", [x, y]));
                } else if (methodName === "GetScreenSize") {
                    const monitor = global.display.get_monitor_geometry(0);
                    invocation.return_value(
                        new GLib.Variant("(ii)", [monitor.width, monitor.height])
                    );
                } else {
                    invocation.return_dbus_error(
                        "org.ai4.Error.UnknownMethod",
                        `Unknown method: ${methodName}`
                    );
                }
            },
            null,
            null
        );

        this._nameId = Gio.DBus.session.own_name(
            "org.ai4.CursorPosition",
            Gio.BusNameOwnerFlags.NONE,
            null,
            null
        );

        console.log("[AI4 CursorPosition] Extension enabled — DBus service registered.");
    }

    disable() {
        if (this._dbusId) {
            Gio.DBus.session.unregister_object(this._dbusId);
            this._dbusId = null;
        }
        if (this._nameId) {
            Gio.DBus.session.unown_name(this._nameId);
            this._nameId = null;
        }
        console.log("[AI4 CursorPosition] Extension disabled.");
    }
}
